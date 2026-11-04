import Foundation
import Observation
import os
import SustitutoSeguroCore

/// El modelo del perfil (`modelo_perfil.json`) y las recomendaciones precalculadas
/// (`recomendaciones.json`, las mismas de la web), cargados desde el paquete de la app. Todo corre en el
/// teléfono: no hay servidor.
///
/// `@MainActor`: SwiftUI solo se entera de los cambios de un objeto observable si
/// ocurren en el hilo principal. La lectura y decodificación de los JSON corre
/// aparte (`Task.detached`) y solo la asignación final vuelve al hilo principal.
@MainActor
@Observable
final class Catalogo {
    private(set) var perfilador: Perfilador?
    private(set) var recomendador: Recomendador?
    private(set) var error: String?

    func cargar() async {
        guard perfilador == nil else { return }
        let inicio = Date()
        do {
            let (perfilador, recomendador) = try await Task.detached(priority: .userInitiated) {
                let perfilador = try Perfilador(datos: try Self.recurso("modelo_perfil"))
                let recomendador = try Recomendador(datos: try Self.recurso("recomendaciones"),
                                                    nombresEspanol: perfilador.modelo.nombresEs)
                return (perfilador, recomendador)
            }.value
            self.perfilador = perfilador
            self.recomendador = recomendador
            Self.registro.info("Modelo y recomendaciones cargados en \(Date().timeIntervalSince(inicio), format: .fixed(precision: 2)) s")
        } catch {
            self.error = error.localizedDescription
            Self.registro.error("No se pudieron cargar los datos: \(error.localizedDescription, privacy: .public)")
        }
    }

    private static let registro = Logger(subsystem: "cl.unknumb.sustitutoseguro", category: "catalogo")

    nonisolated private static func recurso(_ nombre: String) throws -> Data {
        guard let url = Bundle.main.url(forResource: nombre, withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: "\(nombre).json"])
        }
        return try Data(contentsOf: url)
    }
}
