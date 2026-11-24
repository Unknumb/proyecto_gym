import SwiftUI
import SustitutoSeguroCore

/// Elige un ejercicio y la articulación lesionada; el sistema propone hasta 5
/// alternativas que no la cargan, o se abstiene y explica por qué. Cada
/// alternativa se puede calificar: esa revisión es un dato para el modelo.
struct SustituirView: View {
    @Environment(Catalogo.self) private var catalogo
    @State private var busqueda = ""
    @State private var ejercicio: Ejercicio?

    var body: some View {
        let recomendador = catalogo.recomendador!
        List {
            if let ejercicio {
                ConsultaSustitucion(ejercicio: ejercicio) { self.ejercicio = nil }
                    .id(ejercicio.id)
            } else if busqueda.isEmpty {
                Section {
                    Text("Busca un ejercicio por su nombre, en español o en inglés: «sentadilla», «press banca», «lunge».")
                        .foregroundStyle(.secondary)
                }
            } else {
                let resultados = recomendador.buscar(busqueda)
                if resultados.isEmpty {
                    ContentUnavailableView.search(text: busqueda)
                }
                ForEach(resultados) { e in
                    Button { ejercicio = e; busqueda = "" } label: { FilaEjercicio(ejercicio: e) }
                        .buttonStyle(.plain)
                }
            }
        }
        .navigationTitle("Sustituir")
        .searchable(text: $busqueda, prompt: "Ejercicio que quieres reemplazar")
    }
}

struct FilaEjercicio: View {
    let ejercicio: Ejercicio

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(Textos.capitalizar(ejercicio.nombreVisible))
            Text("\(Textos.capitalizar(ejercicio.familia)) · \(ejercicio.equipamiento)")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

private struct ConsultaSustitucion: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    let ejercicio: Ejercicio
    let cambiar: () -> Void
    @State private var articulacion = ""
    @State private var principiante = false
    @State private var calificando: Alternativa?

    var body: some View {
        let recomendador = catalogo.recomendador!
        Section("Ejercicio") {
            NavigationLink {
                DetalleEjercicioView(ejercicio: ejercicio)
            } label: {
                FilaEjercicio(ejercicio: ejercicio)
            }
            Button("Cambiar de ejercicio", action: cambiar)
        }
        .sheet(item: $calificando) { alternativa in
            CalificarAlternativa(ejercicio: ejercicio, alternativa: alternativa, articulacion: articulacion,
                                 principiante: principiante)
        }
        Section {
            Picker("Articulación lesionada", selection: $articulacion) {
                Text("Elige una").tag("")
                ForEach(ejercicio.articulaciones, id: \.self) { Text(Textos.capitalizar($0)).tag($0) }
            }
            Toggle("Soy principiante", isOn: $principiante)
        } footer: {
            Text("Solo aparecen las articulaciones que este ejercicio carga. Principiante: complejidad baja, sin saltos ni inestabilidad.")
        }
        .onAppear { if articulacion.isEmpty { articulacion = ejercicio.articulaciones.first ?? "" } }

        if !articulacion.isEmpty {
            let recomendacion = recomendador.recomendar(ejercicio, articulacion: articulacion, principiante: principiante)
            Section {
                if let titulo = recomendacion.motivo.titulo {
                    VStack(alignment: .leading, spacing: 4) {
                        Label(titulo, systemImage: recomendacion.alternativas.isEmpty ? "hand.raised" : "info.circle")
                            .font(.headline)
                        if let detalle = recomendacion.motivo.detalle { Text(detalle).font(.callout) }
                    }
                    .foregroundStyle(recomendacion.alternativas.isEmpty ? Color.red : Color.orange)
                }
                ForEach(recomendacion.alternativas) { alternativa in
                    FilaAlternativa(alternativa: alternativa,
                                    revision: almacen.revision(original: ejercicio.id, alternativa: alternativa.id,
                                                               articulacion: articulacion)) {
                        calificando = alternativa
                    }
                }
            } header: {
                Text("Alternativas que no cargan la \(articulacion)")
            } footer: {
                if !recomendacion.alternativas.isEmpty {
                    Text("Toca «Calificar» para decir si la alternativa sirve. Tus revisiones se exportan para mejorar el modelo.")
                }
            }
        }
    }
}

private struct FilaAlternativa: View {
    let alternativa: Alternativa
    let revision: RevisionRecomendacion?
    let calificar: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            NavigationLink {
                DetalleEjercicioView(ejercicio: alternativa.ejercicio)
            } label: {
                VStack(alignment: .leading, spacing: 2) {
                    Text(Textos.capitalizar(alternativa.ejercicio.nombreVisible))
                    Text("\(alternativa.mismaFamilia ? "Mismo movimiento" : "Otro movimiento") · similitud \(Textos.porcentaje(alternativa.similitud)) · \(alternativa.ejercicio.equipamiento)")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Button(revision?.veredicto.titulo ?? "Calificar", action: calificar)
                .buttonStyle(.bordered)
                .tint(revision.map { $0.veredicto == .inseguro || $0.veredicto == .malo ? Color.red : Color.green })
                .font(.caption)
        }
    }
}

private struct CalificarAlternativa: View {
    @Environment(Almacen.self) private var almacen
    @Environment(\.dismiss) private var cerrar
    let ejercicio: Ejercicio
    let alternativa: Alternativa
    let articulacion: String
    let principiante: Bool
    @State private var veredicto: Veredicto = .aceptable
    @State private var comentario = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent("Original", value: Textos.capitalizar(ejercicio.nombreVisible))
                    LabeledContent("Alternativa", value: Textos.capitalizar(alternativa.ejercicio.nombreVisible))
                    LabeledContent("Articulación vetada", value: articulacion)
                }
                Section {
                    Picker("Veredicto", selection: $veredicto) {
                        ForEach(Veredicto.allCases) { Text($0.titulo).tag($0) }
                    }
                    .pickerStyle(.segmented)
                } footer: {
                    Text("Insegura: carga la articulación lesionada o es riesgosa. Mala: no trabaja lo mismo (por ejemplo, un estiramiento en lugar de un ejercicio de fuerza).")
                }
                Section("Comentario (opcional)") {
                    TextField("Por qué", text: $comentario, axis: .vertical)
                }
            }
            .navigationTitle("Calificar alternativa")
            .tituloEnLinea()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancelar") { cerrar() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Guardar") {
                        almacen.calificar(RevisionRecomendacion(
                            articulacion: articulacion, principiante: principiante, originalId: ejercicio.id,
                            original: ejercicio.nombre, familiaOriginal: ejercicio.familia,
                            alternativaId: alternativa.id, alternativa: alternativa.ejercicio.nombre,
                            familiaAlternativa: alternativa.ejercicio.familia, mismaFamilia: alternativa.mismaFamilia,
                            similitud: alternativa.similitud, veredicto: veredicto,
                            comentario: comentario.trimmingCharacters(in: .whitespacesAndNewlines)))
                        cerrar()
                    }
                }
            }
            .onAppear {
                if let previa = almacen.revision(original: ejercicio.id, alternativa: alternativa.id, articulacion: articulacion) {
                    veredicto = previa.veredicto
                    comentario = previa.comentario
                }
            }
        }
    }
}
