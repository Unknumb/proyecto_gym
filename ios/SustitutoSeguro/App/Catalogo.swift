import Foundation
import Observation
import SustitutoSeguroCore

/// El modelo del perfil (`modelo_perfil.json`) y las recomendaciones precalculadas
/// (`recomendaciones.json`, las mismas de la web), cargados desde el paquete de la app. Todo corre en el
/// teléfono: no hay servidor.
@Observable
final class Catalogo {
    private(set) var perfilador: Perfilador?
    private(set) var recomendador: Recomendador?
    private(set) var error: String?

    func cargar() async {
        guard perfilador == nil else { return }
        do {
            let (perfilador, recomendador) = try await Task.detached(priority: .userInitiated) {
                let perfilador = try Perfilador(datos: try Self.recurso("modelo_perfil"))
                let recomendador = try Recomendador(datos: try Self.recurso("recomendaciones"),
                                                    nombresEspanol: perfilador.modelo.nombresEs)
                return (perfilador, recomendador)
            }.value
            self.perfilador = perfilador
            self.recomendador = recomendador
        } catch {
            self.error = error.localizedDescription
        }
    }

    private static func recurso(_ nombre: String) throws -> Data {
        guard let url = Bundle.main.url(forResource: nombre, withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: "\(nombre).json"])
        }
        return try Data(contentsOf: url)
    }
}
