import SwiftUI
import SustitutoSeguroCore

/// Entrenar: elegir el día de la rutina, marcar cada serie (kilos y repeticiones) con el
/// descanso cronometrado, y al terminar anotar esfuerzo y dolor por zona. Igual que en la web,
/// un dolor de 4/10 o más avisa, y uno de 7/10 o más pide detenerse y consultar.
struct EntrenarView: View {
    @Environment(Almacen.self) private var almacen

    var body: some View {
        Group {
            if let rutina = almacen.contenido.rutina {
                if almacen.contenido.enCurso != nil {
                    SesionEnCursoView(rutina: rutina)
                } else if rutina.dias.isEmpty {
                    ContentUnavailableView("Tu rutina no tiene días de entrenamiento", systemImage: "exclamationmark.triangle",
                                           description: Text("Con tus lesiones y equipamiento no encontramos ejercicios seguros. Revisa tus datos o consulta a un kinesiólogo."))
                } else {
                    ElegirDiaView(rutina: rutina)
                }
            } else {
                ContentUnavailableView {
                    Label("Primero crea tu rutina", systemImage: "list.bullet.clipboard")
                } description: {
                    Text("Con tus datos armamos los días de entrenamiento y aquí podrás seguirlos serie a serie.")
                } actions: {
                    NavigationLink("Crear mi rutina") { RutinaView() }.buttonStyle(.borderedProminent)
                }
            }
        }
        .navigationTitle("Entrenar")
    }
}

private struct ElegirDiaView: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    let rutina: Rutina

    var body: some View {
        let sesiones = almacen.contenido.entrenamientos
        let resumen = Progreso.resumen(sesiones, hoy: Fechas.iso(), objetivo: almacen.contenido.perfil?.dias ?? rutina.dias.count)
        let sugerido = Progreso.diaSugerido(sesiones, dias: rutina.dias.count)
        List {
            Section {
                Text("Llevas \(resumen.estaSemana) de \(resumen.objetivo) sesiones esta semana.").foregroundStyle(.secondary)
            } header: {
                Text("¿Qué día toca hoy?").font(.title2.bold()).foregroundStyle(.primary).textCase(nil)
            }
            ForEach(Array(rutina.dias.enumerated()), id: \.offset) { k, dia in
                Button { almacen.empezar(dia: k) } label: {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Día \(k + 1) · \(dia.titulo)").font(.headline)
                            Spacer()
                            if k == sugerido {
                                Text("Te toca").font(.caption.bold()).padding(.horizontal, 8).padding(.vertical, 3)
                                    .background(.teal.opacity(0.2), in: Capsule()).foregroundStyle(.teal)
                            }
                        }
                        Text(dia.ejercicios.prefix(4).map { catalogo.nombre($0.id) }.joined(separator: " · ") + (dia.ejercicios.count > 4 ? "…" : ""))
                            .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.leading)
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }
}

private struct SesionEnCursoView: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    let rutina: Rutina
    @State private var finDescanso: Date?
    @State private var terminando = false
    @State private var confirmarDescarte = false

    var body: some View {
        if let s = almacen.contenido.enCurso {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Día \(s.dia + 1) · \(s.titulo)").font(.title3.bold())
                        ProgressView(value: Double(s.seriesHechas), total: Double(max(1, s.totalSeries)))
                        Text("\(s.seriesHechas) de \(s.totalSeries) series").font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    }
                }
                ForEach(Array(s.ejercicios.enumerated()), id: \.offset) { k, e in
                    Section {
                        HStack {
                            Text("Serie").frame(width: 44, alignment: .leading)
                            Text(e.iso ? "—" : "Kg").frame(maxWidth: .infinity)
                            Text(e.iso ? "Segundos" : "Reps").frame(maxWidth: .infinity)
                            Text("✓").frame(width: 44)
                        }
                        .font(.caption).foregroundStyle(.secondary)
                        ForEach(Array(e.series.enumerated()), id: \.offset) { n, serie in
                            FilaSerie(numero: n + 1, iso: e.iso, serie: serie) { nueva in
                                let antes = serie.hecho
                                almacen.contenido.enCurso?.ejercicios[k].series[n] = nueva
                                if nueva.hecho && !antes { finDescanso = Date().addingTimeInterval(TimeInterval(e.descanso)) }
                            }
                        }
                    } header: {
                        HStack {
                            Text(catalogo.nombre(e.id)).textCase(nil).font(.headline).foregroundStyle(.primary)
                            Spacer()
                            Text(e.objetivo).monospacedDigit()
                        }
                    }
                }
                Section {
                    Button("Terminar sesión") { finDescanso = nil; terminando = true }.bold()
                    Button("Descartar", role: .destructive) { confirmarDescarte = true }
                }
            }
            .safeAreaInset(edge: .bottom) {
                if let fin = finDescanso { Descanso(fin: fin, sumar: { finDescanso = fin.addingTimeInterval(30) }, saltar: { finDescanso = nil }) }
            }
            .sheet(isPresented: $terminando) { FinalizarView() }
            .confirmationDialog("¿Descartar esta sesión?", isPresented: $confirmarDescarte, titleVisibility: .visible) {
                Button("Sí, descartar", role: .destructive) { finDescanso = nil; almacen.contenido.enCurso = nil }
            } message: {
                Text("No se guardará nada.")
            }
        }
    }
}

private struct FilaSerie: View {
    let numero: Int
    let iso: Bool
    let serie: SerieEnCurso
    let cambiar: (SerieEnCurso) -> Void

    var body: some View {
        HStack {
            Text("\(numero)").frame(width: 44, alignment: .leading).monospacedDigit()
            TextField(iso ? "—" : "0", value: Binding(get: { serie.kg }, set: { v in var s = serie; s.kg = v.map { max(0, $0) }; cambiar(s) }),
                      format: .number)
                .tecladoDecimal().multilineTextAlignment(.center).disabled(iso)
                .textFieldStyle(.roundedBorder)
                .accessibilityLabel("Kilos serie \(numero)")
            TextField("0", value: Binding(get: { serie.reps }, set: { v in var s = serie; s.reps = max(0, v); cambiar(s) }),
                      format: .number)
                .tecladoNumerico().multilineTextAlignment(.center)
                .textFieldStyle(.roundedBorder)
                .accessibilityLabel("Repeticiones serie \(numero)")
            Button {
                var s = serie; s.hecho.toggle(); cambiar(s)
            } label: {
                Image(systemName: serie.hecho ? "checkmark.circle.fill" : "circle")
                    .font(.title2).foregroundStyle(serie.hecho ? .green : .secondary)
            }
            .buttonStyle(.plain)
            .frame(width: 44)
            .accessibilityLabel(serie.hecho ? "Serie \(numero) hecha" : "Marcar serie \(numero) como hecha")
        }
        .opacity(serie.hecho ? 0.7 : 1)
    }
}

/// El cronómetro de descanso, abajo de la pantalla.
private struct Descanso: View {
    let fin: Date
    let sumar: () -> Void
    let saltar: () -> Void

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { contexto in
            let resto = max(0, Int(fin.timeIntervalSince(contexto.date).rounded()))
            HStack {
                Text("Descanso").font(.subheadline)
                Spacer()
                Text("\(resto / 60):\(String(format: "%02d", resto % 60))").font(.title2.monospacedDigit().bold())
                Spacer()
                Button("+30 s", action: sumar).buttonStyle(.bordered)
                Button("Saltar", action: saltar).buttonStyle(.bordered)
            }
            .padding()
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
            .padding(.horizontal)
            .onChange(of: resto == 0) { _, terminado in if terminado { saltar() } }
            .sensoryFeedback(.success, trigger: resto == 0)
        }
    }
}

private struct FinalizarView: View {
    @Environment(Almacen.self) private var almacen
    @Environment(\.dismiss) private var cerrar

    var body: some View {
        NavigationStack {
            if let s = almacen.contenido.enCurso {
                Form {
                    Section {
                        VStack(alignment: .leading) {
                            Text("Esfuerzo: \(s.esfuerzo)/10").bold()
                            Slider(value: Binding(get: { Double(s.esfuerzo) }, set: { almacen.contenido.enCurso?.esfuerzo = Int($0.rounded()) }),
                                   in: 1...10, step: 1)
                            Text("1 = muy fácil · 10 = no podía hacer ni una repetición más.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if !s.dolor.isEmpty {
                        Section("Dolor durante o después (0 = nada, 10 = máximo)") {
                            ForEach(TablasRutina.lesiones.map(\.clave).filter { s.dolor[$0] != nil }, id: \.self) { clave in
                                VStack(alignment: .leading) {
                                    Text("\(TablasRutina.nombreLesion(clave)): \(s.dolor[clave] ?? 0)/10")
                                    Slider(value: Binding(get: { Double(s.dolor[clave] ?? 0) },
                                                          set: { almacen.contenido.enCurso?.dolor[clave] = Int($0.rounded()) }),
                                           in: 0...10, step: 1)
                                        .tint((s.dolor[clave] ?? 0) >= 7 ? .red : (s.dolor[clave] ?? 0) >= 4 ? .orange : .teal)
                                }
                            }
                        }
                    }
                    if let alerta = s.alertaDolor { Section { AvisoView(aviso: alerta) } }
                    Section("Notas (opcional)") {
                        TextField("Ej: subí peso en sentadilla, molestia leve al final",
                                  text: Binding(get: { s.notas }, set: { almacen.contenido.enCurso?.notas = String($0.prefix(500)) }),
                                  axis: .vertical)
                    }
                    Section {
                        Button("Guardar sesión") { almacen.terminarSesion(); cerrar() }.bold()
                    } footer: {
                        Text("Se guardan las series marcadas como hechas.")
                    }
                }
                .navigationTitle("¿Cómo te fue?")
                .tituloEnLinea()
                .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Volver a la sesión") { cerrar() } } }
            }
        }
    }
}
