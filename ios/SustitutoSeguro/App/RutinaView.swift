import SwiftUI
import SustitutoSeguroCore

/// Mi rutina: el formulario («Cuéntanos de ti») y la rutina semanal que arma el generador,
/// el mismo de la página. Cada ejercicio se puede cambiar por otra opción segura.
struct RutinaView: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    @State private var editando = false

    var body: some View {
        Group {
            if let rutina = almacen.contenido.rutina, !editando {
                PlanView(rutina: rutina, editar: { editando = true })
            } else {
                PerfilFormView(inicial: almacen.contenido.perfil, cancelar: almacen.contenido.rutina == nil ? nil : { editando = false }) { p in
                    almacen.guardarPerfil(p, generador: catalogo.generador!)
                    editando = false
                }
            }
        }
        .navigationTitle("Mi rutina")
    }
}

// MARK: - Avisos

struct AvisoView: View {
    let aviso: Aviso

    var color: Color { aviso.color == "red" ? .red : aviso.color == "teal" ? .teal : .orange }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: aviso.color == "red" ? "exclamationmark.octagon.fill" : aviso.color == "teal" ? "info.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(color)
            (Text(aviso.titulo).bold() + Text(" " + aviso.texto))
                .font(.subheadline)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - La rutina

private struct PlanView: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    let rutina: Rutina
    let editar: () -> Void
    @State private var detalle: Ejercicio?

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(TablasRutina.nombreTipo(rutina.tipo ?? "full")).font(.title3.bold())
                    Text(rutina.porqueTipo).font(.subheadline).foregroundStyle(.secondary)
                    if let p = rutina.perfil {
                        Text("\(p.dias) días · \(p.minutos) min · \(TablasRutina.nombreObjetivo(rutina.objetivo))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                ForEach(Array(rutina.avisos.enumerated()), id: \.offset) { _, a in AvisoView(aviso: a) }
                if !rutina.huecos.isEmpty {
                    AvisoView(aviso: Aviso("amber", "Sin ejercicio seguro para:",
                                           rutina.huecos.map { (TablasRutina.nombreGrupo[$0] ?? $0).lowercased() }.joined(separator: ", ") + "."))
                }
            }

            ForEach(Array(rutina.dias.enumerated()), id: \.offset) { d, dia in
                Section {
                    if dia.equilibrio {
                        Label("Suma 5 minutos de equilibrio: pararte en un pie junto a una pared, 3 × 20–30 s por lado.", systemImage: "figure.stand")
                            .font(.caption).foregroundStyle(.teal)
                    }
                    ForEach(Array(dia.ejercicios.enumerated()), id: \.offset) { k, e in
                        FilaRutina(item: e)
                            .contentShape(Rectangle())
                            .onTapGesture { detalle = catalogo.recomendador?.ejercicio(id: e.id) }
                            .swipeActions {
                                if !e.alternativas.isEmpty {
                                    Button("Cambiar") { almacen.cambiarEjercicio(dia: d, posicion: k) }.tint(.teal)
                                }
                            }
                            .contextMenu {
                                if !e.alternativas.isEmpty {
                                    Button("Cambiar por otra opción segura", systemImage: "arrow.triangle.2.circlepath") {
                                        almacen.cambiarEjercicio(dia: d, posicion: k)
                                    }
                                }
                            }
                    }
                } header: {
                    Text("Día \(d + 1) · \(dia.titulo)")
                } footer: {
                    if d == 0 { Text("Desliza un ejercicio a la izquierda para cambiarlo por otra opción segura.") }
                }
            }

            Section {
                NavigationLink { EntrenarView() } label: { Label("Empezar a entrenar", systemImage: "play.fill") }
                Button("Cambiar mis datos", systemImage: "slider.horizontal.3", action: editar)
                Button("Generar otra variante", systemImage: "shuffle") { almacen.otraVariante(generador: catalogo.generador!) }
            } footer: {
                Text("Creada el \(Fechas.larga(rutina.creada)). Ningún ejercicio carga tus lesiones; si algo duele, detente.")
            }
        }
        .sheet(item: $detalle) { e in NavigationStack { DetalleEjercicioView(ejercicio: e) } }
    }
}

private struct FilaRutina: View {
    @Environment(Catalogo.self) private var catalogo
    let item: ItemRutina

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(catalogo.nombre(item.id))
                Text(TablasRutina.nombreGrupo[item.grupo] ?? item.grupo).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text("\(item.series) × \(item.reps)").monospacedDigit().bold()
                Text("descanso \(Self.descanso(item.descanso))").font(.caption).foregroundStyle(.secondary)
            }
            if item.extra {
                Image(systemName: "sportscourt").foregroundStyle(.orange).accessibilityLabel("Complemento de tu deporte")
            }
        }
    }

    static func descanso(_ s: Int) -> String {
        s >= 60 ? (s % 60 == 0 ? "\(s / 60) min" : "\(s / 60):\(String(format: "%02d", s % 60)) min") : "\(s) s"
    }
}

// MARK: - Formulario

struct PerfilFormView: View {
    let cancelar: (() -> Void)?
    let guardar: (Perfil) -> Void
    @State private var edad: Int?
    @State private var peso: Double?
    @State private var altura: Double?
    @State private var p: Perfil
    @State private var error: String?

    init(inicial: Perfil?, cancelar: (() -> Void)?, guardar: @escaping (Perfil) -> Void) {
        self.cancelar = cancelar
        self.guardar = guardar
        let base = inicial ?? Perfil(edad: 0)
        _p = State(initialValue: base)
        _edad = State(initialValue: inicial?.edad)
        _peso = State(initialValue: inicial?.peso)
        _altura = State(initialValue: inicial?.altura)
    }

    var body: some View {
        Form {
            Section {
                TextField("Edad", value: $edad, format: .number).tecladoNumerico()
                TextField("Peso en kg (opcional)", value: $peso, format: .number).tecladoDecimal()
                TextField("Altura en cm (opcional)", value: $altura, format: .number).tecladoNumerico()
                Picker("Deporte que practicas", selection: $p.deporte) {
                    ForEach(TablasRutina.deportes, id: \.clave) { Text($0.nombre).tag($0.clave) }
                }
            } header: {
                Text("Tus datos")
            } footer: {
                if let error { Text(error).foregroundStyle(.red) }
            }

            Section {
                Picker("Nivel", selection: $p.nivel) {
                    Text("Principiante").tag("principiante")
                    Text("Intermedio").tag("intermedio")
                    Text("Avanzado").tag("avanzado")
                }
                Picker("Objetivo", selection: $p.objetivo) {
                    ForEach(TablasRutina.objetivos, id: \.clave) { Text($0.nombre).tag($0.clave) }
                }
                Stepper("Días por semana: \(p.dias)", value: $p.dias, in: 2...6)
                Picker("Minutos por sesión", selection: $p.minutos) {
                    ForEach([30, 45, 60, 75], id: \.self) { Text("\($0) min").tag($0) }
                }
                Picker("Tipo de rutina", selection: $p.tipo) {
                    ForEach(TablasRutina.tipos, id: \.clave) { Text($0.nombre).tag($0.clave) }
                }
            } header: {
                Text("Tu entrenamiento")
            } footer: {
                Text((TablasRutina.tipos.first { $0.clave == p.tipo }?.detalle ?? "") + " Principiante: menos de 6 meses entrenando con constancia.")
            }

            Section {
                ForEach(TablasRutina.equipos.filter { $0.clave != "pc" }, id: \.clave) { eq in
                    Toggle(eq.nombre, isOn: marcado(\.equipo, eq.clave))
                        .disabled((edad ?? 18) < 14 && eq.clave != "banda")
                }
            } header: {
                Text("Equipamiento que tienes")
            } footer: {
                Text((edad ?? 18) < 14 ? "Con menos de 14 años usamos peso corporal y bandas." : "El peso corporal siempre está incluido.")
            }

            Section {
                ForEach(TablasRutina.lesiones, id: \.clave) { l in
                    Toggle(l.nombre, isOn: marcado(\.lesiones, l.clave))
                }
            } header: {
                Text("Lesiones o molestias")
            } footer: {
                Text("Ningún ejercicio de la rutina cargará las zonas que marques. No reemplaza la indicación de un kinesiólogo.")
            }

            Section {
                Button(cancelar == nil ? "Crear mi rutina" : "Guardar y rehacer la rutina") { enviar() }
                    .bold()
                if let cancelar { Button("Cancelar", role: .cancel, action: cancelar) }
            } footer: {
                Text("Tus datos se guardan en este teléfono y, si tienes cuenta, en tu cuenta.")
            }
        }
    }

    private func marcado(_ campo: WritableKeyPath<Perfil, [String]>, _ clave: String) -> Binding<Bool> {
        Binding(get: { p[keyPath: campo].contains(clave) },
                set: { activo in
                    p[keyPath: campo].removeAll { $0 == clave }
                    if activo { p[keyPath: campo].append(clave) }
                })
    }

    private func enviar() {
        guard let edad, (10...100).contains(edad) else { error = "Escribe tu edad (entre 10 y 100 años)."; return }
        if let peso, !(30...250).contains(peso) { error = "El peso debe estar entre 30 y 250 kg."; return }
        if let altura, !(120...230).contains(altura) { error = "La altura debe estar entre 120 y 230 cm."; return }
        error = nil
        var final = p
        final.edad = edad
        final.peso = peso
        final.altura = altura
        guardar(final)
    }
}
