import SwiftUI
import SustitutoSeguroCore

/// Todo lo registrado en el teléfono, y su exportación a CSV para entregárselo al
/// pipeline (`python scripts/importar_datos_ios.py <carpeta>`).
struct MisDatosView: View {
    @Environment(Almacen.self) private var almacen
    @State private var archivos: [URL] = []
    @State private var errorExportar: String?
    @State private var confirmarBorrado = false

    var body: some View {
        let c = almacen.contenido
        List {
            Section {
                LabeledContent("Revisiones de recomendaciones", value: "\(c.revisiones.count)")
                LabeledContent("Etiquetas corregidas", value: "\(c.correcciones.count)")
                LabeledContent("Ejercicios nuevos", value: "\(c.ejerciciosNuevos.count)")
                LabeledContent("Sesiones de entrenamiento", value: "\(c.sesiones.count)")
            } footer: {
                Text("Los datos viven solo en este teléfono. Salen de él únicamente cuando los exportas.")
            }

            Section {
                Button("Preparar los CSV") {
                    do { archivos = try almacen.exportar(); errorExportar = nil }
                    catch { errorExportar = error.localizedDescription }
                }
                .disabled(almacen.total == 0)
                if !archivos.isEmpty {
                    ShareLink(items: archivos) {
                        Label("Compartir \(archivos.count) archivos", systemImage: "square.and.arrow.up")
                    }
                }
                if let errorExportar { Text(errorExportar).foregroundStyle(.red) }
                if let error = almacen.errorAlGuardar { Text("No se pudo guardar: \(error)").foregroundStyle(.red) }
            } header: {
                Text("Entregar los datos al modelo")
            } footer: {
                Text("Envíalos al computador (AirDrop o Archivos) y corre «python scripts/importar_datos_ios.py <carpeta>» en el repositorio.")
            }
            .onChange(of: almacen.total) { archivos = [] }

            if !c.revisiones.isEmpty {
                Section("Revisiones") {
                    ForEach(c.revisiones.reversed()) { r in
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(r.veredicto.titulo): \(r.alternativa)")
                            Text("en lugar de \(r.original) · \(r.articulacion) vetada").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .onDelete { borrar(\.revisiones, $0) }
                }
            }
            if !c.correcciones.isEmpty {
                Section("Correcciones") {
                    ForEach(c.correcciones.reversed()) { k in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(k.nombre)
                            Text("\(Textos.objetivo[k.objetivo] ?? k.objetivo): \(Textos.clase(k.etiquetaPipeline)) → \(Textos.clase(k.etiquetaCorregida))")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .onDelete { borrar(\.correcciones, $0) }
                }
            }
            if !c.ejerciciosNuevos.isEmpty {
                Section("Ejercicios nuevos") {
                    ForEach(c.ejerciciosNuevos.reversed()) { e in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(e.nombre)
                            Text(Perfilador.objetivos.map { Textos.clase(e.etiquetas[$0]?.final) }.joined(separator: " · "))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .onDelete { borrar(\.ejerciciosNuevos, $0) }
                }
            }

            Section {
                Button("Borrar todos los datos", role: .destructive) { confirmarBorrado = true }
                    .disabled(almacen.total == 0)
            }
        }
        .navigationTitle("Mis datos")
        .confirmationDialog("¿Borrar todos los datos del teléfono?", isPresented: $confirmarBorrado,
                            titleVisibility: .visible) {
            Button("Borrar todo", role: .destructive) { almacen.borrarTodo() }
        } message: {
            Text("No se puede deshacer. Exporta antes si quieres conservarlos.")
        }
    }

    /// Las listas se muestran de la más nueva a la más antigua: el índice visible se
    /// traduce al de la lista guardada.
    private func borrar<T>(_ ruta: WritableKeyPath<Almacen.Contenido, [T]>, _ indices: IndexSet) {
        let n = almacen.contenido[keyPath: ruta].count
        almacen.contenido[keyPath: ruta].remove(atOffsets: IndexSet(indices.map { n - 1 - $0 }))
    }
}
