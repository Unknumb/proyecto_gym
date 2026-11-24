import SwiftUI
import SustitutoSeguroCore

/// Registro de entrenamiento: series con peso, repeticiones y esfuerzo, y el
/// dolor por zona al terminar. Igual que en la web, un dolor de 7/10 o más avisa.
struct EntrenarView: View {
    @Environment(Almacen.self) private var almacen
    @State private var nueva = false

    var body: some View {
        let sesiones = almacen.contenido.sesiones.sorted { $0.fecha > $1.fecha }
        List {
            if sesiones.isEmpty {
                ContentUnavailableView("Sin entrenamientos", systemImage: "figure.strengthtraining.traditional",
                                       description: Text("Registra tu primera sesión con el botón +."))
            }
            ForEach(sesiones) { sesion in
                NavigationLink {
                    SesionView(sesion: sesion)
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(sesion.fecha.formatted(date: .abbreviated, time: .shortened))
                        Text(resumen(sesion)).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .onDelete { indices in
                let ids = Set(indices.map { sesiones[$0].id })
                almacen.contenido.sesiones.removeAll { ids.contains($0.id) }
            }
        }
        .navigationTitle("Entrenar")
        .toolbar {
            Button { nueva = true } label: { Label("Nueva sesión", systemImage: "plus") }
        }
        .sheet(isPresented: $nueva) { NuevaSesionView() }
    }

    private func resumen(_ s: SesionEntrenamiento) -> String {
        let ejercicios = Set(s.series.map(\.ejercicioId)).count
        let maximo = s.dolores.map(\.dolor).max()
        return "\(s.series.count) series de \(ejercicios) ejercicios" + (maximo.map { " · dolor máximo \($0)/10" } ?? "")
    }
}

private struct SesionView: View {
    let sesion: SesionEntrenamiento

    var body: some View {
        List {
            Section("Series") {
                ForEach(sesion.series) { s in
                    LabeledContent(Textos.capitalizar(s.nombre),
                                   value: "\(s.pesoKg.formatted()) kg × \(s.repeticiones) · esfuerzo \(s.esfuerzo)/10")
                }
            }
            if !sesion.dolores.isEmpty {
                Section("Dolor por zona") {
                    ForEach(sesion.dolores) { d in
                        LabeledContent(Textos.capitalizar(d.zona), value: "\(d.dolor)/10")
                            .foregroundStyle(d.dolor >= 7 ? Color.red : Color.primary)
                    }
                }
            }
            if !sesion.notas.isEmpty { Section("Notas") { Text(sesion.notas) } }
        }
        .navigationTitle(sesion.fecha.formatted(date: .abbreviated, time: .omitted))
    }
}

private struct NuevaSesionView: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    @Environment(\.dismiss) private var cerrar
    @State private var sesion = SesionEntrenamiento()
    @State private var eligiendo = false

    var body: some View {
        NavigationStack {
            Form {
                DatePicker("Fecha", selection: $sesion.fecha)
                Section("Series") {
                    ForEach($sesion.series) { $serie in
                        VStack(alignment: .leading) {
                            Text(Textos.capitalizar(serie.nombre)).font(.subheadline)
                            HStack {
                                TextField("kg", value: $serie.pesoKg, format: .number).tecladoDecimal()
                                Text("kg ×").foregroundStyle(.secondary)
                                TextField("reps", value: $serie.repeticiones, format: .number).tecladoNumerico()
                                Text("reps").foregroundStyle(.secondary)
                            }
                            Stepper("Esfuerzo \(serie.esfuerzo)/10", value: $serie.esfuerzo, in: 1...10)
                        }
                    }
                    .onDelete { sesion.series.remove(atOffsets: $0) }
                    Button("Agregar serie") { eligiendo = true }
                    if let ultima = sesion.series.last {
                        Button("Repetir la última serie") {
                            var copia = ultima
                            copia.id = UUID()
                            sesion.series.append(copia)
                        }
                    }
                }
                Section {
                    ForEach($sesion.dolores) { $dolor in
                        Stepper("\(Textos.capitalizar(dolor.zona)): \(dolor.dolor)/10", value: $dolor.dolor, in: 0...10)
                    }
                    .onDelete { sesion.dolores.remove(atOffsets: $0) }
                    Menu("Agregar zona") {
                        ForEach(catalogo.recomendador!.articulaciones.filter { zona in
                            !sesion.dolores.contains { $0.zona == zona }
                        }, id: \.self) { zona in
                            Button(Textos.capitalizar(zona)) { sesion.dolores.append(RegistroDolor(zona: zona, dolor: 0)) }
                        }
                    }
                } header: {
                    Text("Dolor al terminar")
                } footer: {
                    if sesion.dolores.contains(where: { $0.dolor >= 7 }) {
                        Text("Un dolor de 7/10 o más es una señal para detenerse y consultar a un kinesiólogo.")
                            .foregroundStyle(.red)
                    }
                }
                Section("Notas") { TextField("Opcional", text: $sesion.notas, axis: .vertical) }
            }
            .navigationTitle("Nueva sesión")
            .tituloEnLinea()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancelar") { cerrar() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Guardar") {
                        almacen.contenido.sesiones.append(sesion)
                        cerrar()
                    }
                    .disabled(sesion.series.isEmpty && sesion.dolores.isEmpty)
                }
            }
            .sheet(isPresented: $eligiendo) {
                BuscadorEjercicio { e in
                    sesion.series.append(SerieEntrenamiento(ejercicioId: e.id, nombre: e.nombreVisible,
                                                            pesoKg: 0, repeticiones: 10, esfuerzo: 6))
                }
            }
        }
    }
}

/// Hoja para elegir un ejercicio del catálogo por nombre.
struct BuscadorEjercicio: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(\.dismiss) private var cerrar
    let elegir: (Ejercicio) -> Void
    @State private var busqueda = ""

    var body: some View {
        NavigationStack {
            List(catalogo.recomendador!.buscar(busqueda)) { e in
                Button { elegir(e); cerrar() } label: { FilaEjercicio(ejercicio: e) }
                    .buttonStyle(.plain)
            }
            .overlay {
                if busqueda.isEmpty {
                    ContentUnavailableView("Busca el ejercicio", systemImage: "magnifyingglass",
                                           description: Text("En español o en inglés."))
                }
            }
            .searchable(text: $busqueda, prompt: "Ejercicio")
            .navigationTitle("Elegir ejercicio")
            .tituloEnLinea()
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancelar") { cerrar() } } }
        }
    }
}
