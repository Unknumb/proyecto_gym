import SwiftUI
import SustitutoSeguroCore

/// Lo que el sistema sabe de un ejercicio del grafo y, para cada etiqueta del
/// perfil, de dónde viene y con qué confianza. Cada etiqueta se puede corregir:
/// la corrección es un dato para reentrenar el modelo.
struct DetalleEjercicioView: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    let ejercicio: Ejercicio
    @State private var corrigiendo: String?

    var body: some View {
        let perfilador = catalogo.perfilador!
        List {
            Section {
                if ejercicio.nombreEspanol != nil { LabeledContent("Nombre original", value: ejercicio.nombre) }
                LabeledContent("Familia de movimiento", value: Textos.capitalizar(ejercicio.familia))
                LabeledContent("Equipamiento", value: ejercicio.equipamiento)
                LabeledContent("Músculos principales", value: ejercicio.agonistas.joined(separator: ", "))
                LabeledContent("Articulaciones que carga", value: ejercicio.articulaciones.joined(separator: ", "))
                LabeledContent("Fuente", value: ejercicio.fuente)
            }
            Section {
                ForEach(Perfilador.objetivos, id: \.self) { objetivo in
                    let guardado = perfilador.perfilGuardado(ejercicioId: ejercicio.id, objetivo: objetivo)
                    let correccion = almacen.correccion(ejercicioId: ejercicio.id, objetivo: objetivo)
                    Button { corrigiendo = objetivo } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text(Textos.objetivo[objetivo] ?? objetivo)
                                Spacer()
                                Text(Textos.clase(guardado?.etiqueta)).foregroundStyle(.secondary)
                                if let correccion {
                                    Image(systemName: "arrow.right")
                                    Text(Textos.clase(correccion.etiquetaCorregida)).bold()
                                }
                            }
                            Text(descripcion(guardado)).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                }
            } header: {
                Text("Perfil de exigencia")
            } footer: {
                Text("Toca una etiqueta para corregirla si está mal. «Sin etiqueta»: el modelo no tuvo confianza suficiente y prefirió abstenerse.")
            }
        }
        .navigationTitle(Textos.capitalizar(ejercicio.nombreVisible))
        .tituloEnLinea()
        .sheet(item: Binding(get: { corrigiendo.map(ObjetivoElegido.init) }, set: { corrigiendo = $0?.id })) { elegido in
            CorregirEtiqueta(ejercicio: ejercicio, objetivo: elegido.id)
        }
    }

    private func descripcion(_ guardado: PerfilGuardado?) -> String {
        guard let guardado, guardado.etiqueta != nil else { return "El modelo se abstuvo" }
        let origen = Textos.origen[guardado.origen ?? ""] ?? "origen desconocido"
        guard guardado.origen == "modelo", let confianza = guardado.confianza else { return Textos.capitalizar(origen) }
        return "\(Textos.capitalizar(origen)), confianza \(Textos.porcentaje(confianza))"
    }
}

private struct ObjetivoElegido: Identifiable { let id: String }

private struct CorregirEtiqueta: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    @Environment(\.dismiss) private var cerrar
    let ejercicio: Ejercicio
    let objetivo: String
    @State private var etiqueta = ""
    @State private var comentario = ""

    var body: some View {
        let perfilador = catalogo.perfilador!
        let guardado = perfilador.perfilGuardado(ejercicioId: ejercicio.id, objetivo: objetivo)
        NavigationStack {
            Form {
                Section {
                    LabeledContent("Ejercicio", value: Textos.capitalizar(ejercicio.nombreVisible))
                    LabeledContent("Etiqueta actual", value: Textos.clase(guardado?.etiqueta))
                }
                Section("Etiqueta correcta") {
                    Picker(Textos.objetivo[objetivo] ?? objetivo, selection: $etiqueta) {
                        ForEach(perfilador.clases(objetivo), id: \.self) { Text(Textos.clase($0)).tag($0) }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                }
                Section("Comentario (opcional)") {
                    TextField("Por qué", text: $comentario, axis: .vertical)
                }
            }
            .navigationTitle(Textos.objetivo[objetivo] ?? objetivo)
            .tituloEnLinea()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancelar") { cerrar() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Guardar") {
                        almacen.corregir(CorreccionEtiqueta(
                            ejercicioId: ejercicio.id, nombre: ejercicio.nombre, objetivo: objetivo,
                            etiquetaPipeline: guardado?.etiqueta, origenPipeline: guardado?.origen,
                            confianzaPipeline: guardado?.confianza, etiquetaCorregida: etiqueta,
                            comentario: comentario.trimmingCharacters(in: .whitespacesAndNewlines)))
                        cerrar()
                    }
                    .disabled(etiqueta.isEmpty)
                }
            }
            .onAppear {
                let previa = almacen.correccion(ejercicioId: ejercicio.id, objetivo: objetivo)
                etiqueta = previa?.etiquetaCorregida ?? guardado?.etiqueta ?? ""
                comentario = previa?.comentario ?? ""
            }
        }
    }
}
