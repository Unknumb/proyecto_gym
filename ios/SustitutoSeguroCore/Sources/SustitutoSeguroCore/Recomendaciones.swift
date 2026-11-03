import Foundation

/// Un ejercicio del grafo, como lo exporta `scripts/exportar_datos_web.py`.
public struct Ejercicio: Identifiable, Hashable {
    public let indice: Int
    public let id: String                 // ejercicio_id del pipeline
    public let nombre: String
    public let nombreEspanol: String?
    public let fuente: String
    public let familia: String
    public let equipamiento: String
    public let agonistas: [String]
    public let articulaciones: [String]   // las que carga
    public let complejidad: String?
    public let fuerza: String?
    public let esEstiramiento: Bool
    public let aptoPrincipiante: Bool

    public var nombreVisible: String { nombreEspanol ?? nombre }
}

/// Una alternativa segura propuesta por el `Sustituidor`.
public struct Alternativa: Identifiable, Hashable {
    public let ejercicio: Ejercicio
    public let similitud: Double
    public let mismaFamilia: Bool
    public var id: String { ejercicio.id }
}

/// Por qué el sistema se abstiene o recomienda menos de lo pedido (códigos `MOTIVO`
/// de `exportar_datos_web.py`, con los mismos textos que la web).
public enum Motivo: Int {
    case ninguno = 0, menosDeTres, sinSustitutoAnatomico, musculoConPocosEjercicios, noCargaLaArticulacion

    public var titulo: String? {
        switch self {
        case .ninguno: nil
        case .menosDeTres: "Hay menos de 3 alternativas seguras."
        case .sinSustitutoAnatomico: "No hay un sustituto seguro."
        case .musculoConPocosEjercicios: "Preferimos no recomendar."
        case .noCargaLaArticulacion: "Este ejercicio no carga esa articulación."
        }
    }

    public var detalle: String? {
        switch self {
        case .ninguno: nil
        case .menosDeTres: "Se muestran las que hay."
        case .sinSustitutoAnatomico: "Todos los músculos principales de este ejercicio cruzan la articulación lesionada: cualquier ejercicio que los trabaje también la cargaría. Lo mejor es un plan de rehabilitación con un kinesiólogo."
        case .musculoConPocosEjercicios: "El músculo que habría que conservar tiene menos de 10 ejercicios en el catálogo. Con tan pocas opciones la recomendación no es confiable."
        case .noCargaLaArticulacion: "No hace falta sustituirlo."
        }
    }
}

public struct Recomendacion {
    public let motivo: Motivo
    public let alternativas: [Alternativa]
}

/// Las recomendaciones precalculadas (`recomendaciones.json`, copia de las de la web): las mismas que da el
/// `Sustituidor` del pipeline para cada ejercicio, articulación y perfil (los tests
/// de la web las comparan con el pipeline).
public final class Recomendador {
    public let ejercicios: [Ejercicio]
    public let articulaciones: [String]
    private let rec: [[Int: [(Int, [(Int, Int, Bool)])]]]
    private let indicePorId: [String: Int]

    public init(datos: Data, nombresEspanol: [String: String] = [:]) throws {
        guard let raiz = try JSONSerialization.jsonObject(with: datos) as? [String: Any],
              let ids = raiz["ids"] as? [String], let articulaciones = raiz["articulaciones"] as? [String],
              let familias = raiz["familias"] as? [String], let equipos = raiz["equipos"] as? [String],
              let musculos = raiz["musculos"] as? [String], let filas = raiz["ejercicios"] as? [[Any]],
              let rec = raiz["rec"] as? [[String: [[Any]]]] else {
            throw CocoaError(.coderReadCorrupt)
        }
        self.articulaciones = articulaciones
        ejercicios = filas.enumerated().map { i, f in
            Ejercicio(indice: i, id: ids[i], nombre: f[0] as! String, nombreEspanol: nombresEspanol[ids[i]],
                      fuente: f[1] as! String, familia: familias[f[2] as! Int], equipamiento: equipos[f[3] as! Int],
                      agonistas: (f[4] as! [Int]).map { musculos[$0] },
                      articulaciones: (f[5] as! [Int]).map { articulaciones[$0] },
                      complejidad: (f[6] as? String).flatMap { $0.isEmpty ? nil : $0 },
                      fuerza: (f[7] as? String).flatMap { $0.isEmpty ? nil : $0 },
                      esEstiramiento: f[8] as! Bool, aptoPrincipiante: f[9] as! Bool)
        }
        self.rec = rec.map { porArticulacion in
            Dictionary(uniqueKeysWithValues: porArticulacion.map { clave, perfiles in
                (Int(clave)!, perfiles.map { par in
                    (par[0] as! Int, (par[1] as! [[Int]]).map { ($0[0], $0[1], $0[2] == 1) })
                })
            })
        }
        indicePorId = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
    }

    public func ejercicio(id: String) -> Ejercicio? { indicePorId[id].map { ejercicios[$0] } }

    /// Alternativas para `ejercicio` con `articulacion` lesionada. `principiante`
    /// restringe a ejercicios de complejidad baja, sin impacto ni inestabilidad.
    public func recomendar(_ ejercicio: Ejercicio, articulacion: String, principiante: Bool) -> Recomendacion {
        guard let a = articulaciones.firstIndex(of: articulacion),
              let perfiles = rec[ejercicio.indice][a] else {
            return Recomendacion(motivo: .noCargaLaArticulacion, alternativas: [])
        }
        let (motivo, filas) = perfiles[principiante ? 1 : 0]
        return Recomendacion(motivo: Motivo(rawValue: motivo) ?? .sinSustitutoAnatomico,
                             alternativas: filas.map { j, similitud, misma in
                                 Alternativa(ejercicio: ejercicios[j], similitud: Double(similitud) / 1000,
                                             mismaFamilia: misma)
                             })
    }

    /// Búsqueda sin acentos ni mayúsculas, en el nombre original y en el español.
    public func buscar(_ texto: String, limite: Int = 40) -> [Ejercicio] {
        let consulta = Self.plegar(texto)
        guard !consulta.isEmpty else { return [] }
        let palabras = consulta.split(separator: " ")
        let encontrados = ejercicios.filter { e in
            let nombres = [Self.plegar(e.nombre), e.nombreEspanol.map(Self.plegar) ?? ""]
            return palabras.allSatisfy { p in nombres.contains { $0.contains(p) } }
        }
        // Primero los de nombre en español y los nombres más cortos (los más comunes).
        return Array(encontrados.sorted {
            ($0.nombreEspanol == nil ? 1 : 0, $0.nombre.count) < ($1.nombreEspanol == nil ? 1 : 0, $1.nombre.count)
        }.prefix(limite))
    }

    static func plegar(_ texto: String) -> String {
        texto.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: Locale(identifier: "es"))
    }
}
