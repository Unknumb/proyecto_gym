import Foundation
import Observation
import SustitutoSeguroCore

/// Una calificación de una alternativa propuesta por el sistema.
struct RevisionRecomendacion: Codable, Identifiable, Hashable {
    var id = UUID()
    var fecha = Date()
    var articulacion: String
    var principiante: Bool
    var originalId: String
    var original: String
    var familiaOriginal: String
    var alternativaId: String
    var alternativa: String
    var familiaAlternativa: String
    var mismaFamilia: Bool
    var similitud: Double
    var veredicto: Veredicto
    var comentario: String
}

/// Una etiqueta del perfil que una persona corrige.
struct CorreccionEtiqueta: Codable, Identifiable, Hashable {
    var id = UUID()
    var fecha = Date()
    var ejercicioId: String
    var nombre: String
    var objetivo: String
    var etiquetaPipeline: String?
    var origenPipeline: String?
    var confianzaPipeline: Double?
    var etiquetaCorregida: String
    var comentario: String
}

/// Un ejercicio que no está en el catálogo, con lo que predijo el modelo y la
/// etiqueta final que eligió la persona (`nil` = sin etiqueta).
struct EjercicioNuevo: Codable, Identifiable, Hashable {
    struct Etiqueta: Codable, Hashable {
        var modelo: String
        var confianza: Double
        var final: String?
    }
    var id = UUID()
    var fecha = Date()
    var nombre: String
    var nombreNorm: String
    var instrucciones: String
    var familiaEquipamiento: String
    var musculosPrimarios: [String]
    var musculosSecundarios: [String]
    var etiquetas: [String: Etiqueta]          // por objetivo
    var impacto: Bool
    var estabilidad: String
    var cargaGraduable: Bool
}

struct SerieEntrenamiento: Codable, Identifiable, Hashable {
    var id = UUID()
    var ejercicioId: String
    var nombre: String
    var pesoKg: Double
    var repeticiones: Int
    var esfuerzo: Int                          // 1 a 10
}

struct RegistroDolor: Codable, Identifiable, Hashable {
    var id = UUID()
    var zona: String
    var dolor: Int                             // 0 a 10
}

struct SesionEntrenamiento: Codable, Identifiable, Hashable {
    var id = UUID()
    var fecha = Date()
    var notas = ""
    var series: [SerieEntrenamiento] = []
    var dolores: [RegistroDolor] = []
}

/// Todo lo que la persona registra en la app. Vive solo en el teléfono, en
/// `Documentos/datos_app.json`, y sale de él únicamente cuando se exporta.
@Observable
final class Almacen {
    struct Contenido: Codable {
        var revisiones: [RevisionRecomendacion] = []
        var correcciones: [CorreccionEtiqueta] = []
        var ejerciciosNuevos: [EjercicioNuevo] = []
        var sesiones: [SesionEntrenamiento] = []
    }

    var contenido: Contenido { didSet { guardar() } }
    private(set) var errorAlGuardar: String?
    private let archivo: URL

    init(archivo: URL = URL.documentsDirectory.appending(path: "datos_app.json")) {
        self.archivo = archivo
        let datos = try? Data(contentsOf: archivo)
        contenido = datos.flatMap { try? JSONDecoder().decode(Contenido.self, from: $0) } ?? Contenido()
    }

    private func guardar() {
        do {
            try JSONEncoder().encode(contenido).write(to: archivo, options: [.atomic, .completeFileProtection])
            errorAlGuardar = nil
        } catch {
            errorAlGuardar = error.localizedDescription
        }
    }

    // MARK: Consultas

    func revision(original: String, alternativa: String, articulacion: String) -> RevisionRecomendacion? {
        contenido.revisiones.last {
            $0.originalId == original && $0.alternativaId == alternativa && $0.articulacion == articulacion
        }
    }

    func correccion(ejercicioId: String, objetivo: String) -> CorreccionEtiqueta? {
        contenido.correcciones.last { $0.ejercicioId == ejercicioId && $0.objetivo == objetivo }
    }

    var total: Int {
        contenido.revisiones.count + contenido.correcciones.count + contenido.ejerciciosNuevos.count
            + contenido.sesiones.count
    }

    // MARK: Cambios

    /// Una alternativa tiene una sola calificación vigente: calificarla de nuevo la reemplaza.
    func calificar(_ revision: RevisionRecomendacion) {
        contenido.revisiones.removeAll {
            $0.originalId == revision.originalId && $0.alternativaId == revision.alternativaId
                && $0.articulacion == revision.articulacion
        }
        contenido.revisiones.append(revision)
    }

    func corregir(_ correccion: CorreccionEtiqueta) {
        contenido.correcciones.removeAll { $0.ejercicioId == correccion.ejercicioId && $0.objetivo == correccion.objetivo }
        contenido.correcciones.append(correccion)
    }

    func borrarTodo() { contenido = Contenido() }

    // MARK: Exportación

    /// Escribe los cinco CSV en una carpeta temporal y devuelve sus rutas, para
    /// compartirlos (AirDrop, Archivos, correo) y pasarlos al proyecto con
    /// `python scripts/importar_datos_ios.py <carpeta>`.
    func exportar() throws -> [URL] {
        let carpeta = URL.temporaryDirectory.appending(path: "sustituto_seguro_\(Int(Date().timeIntervalSince1970))")
        try FileManager.default.createDirectory(at: carpeta, withIntermediateDirectories: true)
        return try TablaExportable.allCases.map { tabla in
            let url = carpeta.appending(path: tabla.archivo)
            try CSV.escribir(columnas: tabla.columnas, filas: filas(tabla)).write(to: url, atomically: true, encoding: .utf8)
            return url
        }
    }

    func filas(_ tabla: TablaExportable) -> [[String]] {
        let c = contenido
        switch tabla {
        case .revisiones:
            return c.revisiones.map { r in
                [r.articulacion, r.original, r.familiaOriginal, r.alternativa, r.familiaAlternativa,
                 r.mismaFamilia ? "True" : "False", CSV.numero(r.similitud), r.veredicto.rawValue, r.comentario,
                 r.originalId, r.alternativaId, r.principiante ? "principiante" : "cualquiera", CSV.fecha(r.fecha)]
            }
        case .correcciones:
            return c.correcciones.map { k in
                [k.ejercicioId, k.nombre, k.objetivo, k.etiquetaPipeline ?? "", k.origenPipeline ?? "",
                 CSV.numero(k.confianzaPipeline), k.etiquetaCorregida, k.comentario, CSV.fecha(k.fecha)]
            }
        case .ejerciciosNuevos:
            return c.ejerciciosNuevos.map { e in
                var fila = [e.nombre, e.nombreNorm, e.instrucciones, e.familiaEquipamiento,
                            CSV.lista(e.musculosPrimarios), CSV.lista(e.musculosSecundarios)]
                for objetivo in Perfilador.objetivos {
                    let etiqueta = e.etiquetas[objetivo]
                    fila += [etiqueta?.modelo ?? "", CSV.numero(etiqueta?.confianza), etiqueta?.final ?? ""]
                }
                return fila + [e.impacto ? "True" : "False", e.estabilidad, e.cargaGraduable ? "True" : "False",
                               CSV.fecha(e.fecha)]
            }
        case .series:
            return c.sesiones.flatMap { s in
                s.series.enumerated().map { k, serie in
                    [CSV.fecha(s.fecha), s.id.uuidString, serie.ejercicioId, serie.nombre, String(k + 1),
                     CSV.numero(serie.pesoKg, decimales: 1), String(serie.repeticiones), String(serie.esfuerzo)]
                }
            }
        case .dolor:
            return c.sesiones.flatMap { s in
                s.dolores.map { [CSV.fecha(s.fecha), s.id.uuidString, $0.zona, String($0.dolor)] }
            }
        }
    }
}
