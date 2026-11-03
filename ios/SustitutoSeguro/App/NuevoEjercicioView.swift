import SwiftUI
import SustitutoSeguroCore

/// Un ejercicio que no está en el catálogo: con su nombre, instrucciones,
/// equipamiento y músculos, el modelo predice en el teléfono su mecánica, su tipo
/// de fuerza y su complejidad técnica, o se abstiene. La persona confirma o
/// corrige cada etiqueta y lo guarda como dato nuevo.
struct NuevoEjercicioView: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    @State private var nombre = ""
    @State private var instrucciones = ""
    @State private var familia = "peso libre"
    @State private var primarios: [String] = []
    @State private var secundarios: [String] = []
    @State private var finales: [String: String] = [:]      // objetivo → etiqueta elegida ("" = sin etiqueta)
    @State private var guardado = false

    private var entrada: EntradaEjercicio {
        EntradaEjercicio(nombre: nombre, instrucciones: instrucciones, familiaEquipamiento: familia,
                         musculosPrimarios: primarios, musculosSecundarios: secundarios)
    }

    private var listo: Bool {
        !nombre.trimmingCharacters(in: .whitespaces).isEmpty && !primarios.isEmpty
    }

    var body: some View {
        let perfilador = catalogo.perfilador!
        let ontologia = perfilador.modelo.ontologia
        Form {
            Section {
                TextField("Nombre (en inglés: «dumbbell floor press»)", text: $nombre)
                    .sinMayusculaInicial()
                TextField("Instrucciones (en inglés, opcional)", text: $instrucciones, axis: .vertical)
                    .lineLimit(3...8)
                Picker("Equipamiento", selection: $familia) {
                    ForEach(ontologia.familiasEquipamiento, id: \.self) { Text(Textos.capitalizar($0)).tag($0) }
                }
            } header: {
                Text("Ejercicio")
            } footer: {
                Text("El modelo aprendió con fichas en inglés: el nombre y las instrucciones en inglés dan mejores predicciones.")
            }
            Section("Músculos") {
                NavigationLink {
                    SelectorMusculos(titulo: "Músculos principales", opciones: ontologia.musculos, elegidos: $primarios)
                } label: {
                    LabeledContent("Principales", value: primarios.isEmpty ? "Elegir" : primarios.joined(separator: ", "))
                }
                NavigationLink {
                    SelectorMusculos(titulo: "Músculos secundarios", opciones: ontologia.musculos, elegidos: $secundarios)
                } label: {
                    LabeledContent("Secundarios", value: secundarios.isEmpty ? "Ninguno" : secundarios.joined(separator: ", "))
                }
            }

            if listo {
                let perfil = perfilador.perfil(entrada)
                Section {
                    ForEach(Perfilador.objetivos, id: \.self) { objetivo in
                        let prediccion = perfil.predicciones[objetivo]!
                        VStack(alignment: .leading, spacing: 4) {
                            Picker(Textos.objetivo[objetivo] ?? objetivo, selection: binding(objetivo, prediccion)) {
                                Text("Sin etiqueta").tag("")
                                ForEach(prediccion.clases, id: \.self) { Text(Textos.clase($0)).tag($0) }
                            }
                            Text(prediccion.etiqueta == nil
                                 ? "El modelo se abstiene: confianza \(Textos.porcentaje(prediccion.confianza)) en «\(Textos.clase(prediccion.claseMasProbable))», bajo el umbral de \(Textos.porcentaje(prediccion.umbral))."
                                 : "El modelo predice «\(Textos.clase(prediccion.claseMasProbable))» con confianza \(Textos.porcentaje(prediccion.confianza)).")
                                .font(.caption).foregroundStyle(prediccion.etiqueta == nil ? Color.orange : Color.secondary)
                        }
                    }
                    LabeledContent("Impacto", value: perfil.reglas.impacto ? "Sí (saltos o carrera)" : "No")
                    LabeledContent("Estabilidad", value: Textos.capitalizar(perfil.reglas.estabilidad))
                    LabeledContent("Carga graduable", value: perfil.reglas.cargaGraduable ? "Sí" : "No")
                } header: {
                    Text("Perfil predicho en el teléfono")
                } footer: {
                    Text("Cambia una etiqueta si sabes que es otra: se guarda lo que predijo el modelo y lo que elegiste tú.")
                }
                Section {
                    Button("Guardar ejercicio") { guardar(perfil) }
                }
            }
        }
        .navigationTitle("Nuevo ejercicio")
        .alert("Ejercicio guardado", isPresented: $guardado) {
            Button("Listo", role: .cancel) {}
        } message: {
            Text("Lo verás en Mis datos, listo para exportar.")
        }
    }

    /// La etiqueta elegida; mientras la persona no la toque, la que da el modelo
    /// (o ninguna, si se abstiene).
    private func binding(_ objetivo: String, _ prediccion: Prediccion) -> Binding<String> {
        Binding(get: { finales[objetivo] ?? prediccion.etiqueta ?? "" }, set: { finales[objetivo] = $0 })
    }

    private func guardar(_ perfil: PerfilEjercicio) {
        var etiquetas: [String: EjercicioNuevo.Etiqueta] = [:]
        for (objetivo, prediccion) in perfil.predicciones {
            let elegida = finales[objetivo] ?? prediccion.etiqueta ?? ""
            etiquetas[objetivo] = .init(modelo: prediccion.claseMasProbable, confianza: prediccion.confianza,
                                        final: elegida.isEmpty ? nil : elegida)
        }
        almacen.contenido.ejerciciosNuevos.append(EjercicioNuevo(
            nombre: nombre.trimmingCharacters(in: .whitespaces), nombreNorm: perfil.nombreNormalizado,
            instrucciones: instrucciones.trimmingCharacters(in: .whitespacesAndNewlines),
            familiaEquipamiento: familia, musculosPrimarios: primarios, musculosSecundarios: secundarios,
            etiquetas: etiquetas, impacto: perfil.reglas.impacto, estabilidad: perfil.reglas.estabilidad,
            cargaGraduable: perfil.reglas.cargaGraduable))
        nombre = ""; instrucciones = ""; primarios = []; secundarios = []; finales = [:]
        guardado = true
    }
}

/// Lista de músculos de la ontología común, con marca en los elegidos.
struct SelectorMusculos: View {
    let titulo: String
    let opciones: [String]
    @Binding var elegidos: [String]

    var body: some View {
        List(opciones, id: \.self) { musculo in
            Button {
                if let k = elegidos.firstIndex(of: musculo) { elegidos.remove(at: k) } else { elegidos.append(musculo) }
            } label: {
                HStack {
                    Text(Textos.capitalizar(musculo))
                    Spacer()
                    if elegidos.contains(musculo) { Image(systemName: "checkmark").foregroundStyle(.tint) }
                }
            }
            .buttonStyle(.plain)
        }
        .navigationTitle(titulo)
        .tituloEnLinea()
    }
}
