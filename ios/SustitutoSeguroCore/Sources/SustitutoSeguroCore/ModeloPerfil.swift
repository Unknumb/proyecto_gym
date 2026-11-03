import Foundation

/// Contenido de `modelo_perfil.json`, escrito por `scripts/exportar_modelo_ios.py`.
public struct ModeloPerfil: Decodable {
    public struct Ontologia: Decodable {
        public let musculos: [String]
        public let articulacionesPorMusculo: [String: [String]]
        public let familiasEquipamiento: [String]
        enum CodingKeys: String, CodingKey {
            case musculos, articulacionesPorMusculo = "articulaciones_por_musculo",
                 familiasEquipamiento = "familias_equipamiento"
        }
    }

    struct Reglas: Decodable {
        let impacto, sinImpacto, inestable, apoyado, estiramiento: String
        let familiasInestable, familiasApoyado, familiasCargaGraduable: [String]
        enum CodingKeys: String, CodingKey {
            case impacto, sinImpacto = "sin_impacto", inestable, apoyado, estiramiento,
                 familiasInestable = "familias_inestable", familiasApoyado = "familias_apoyado",
                 familiasCargaGraduable = "familias_carga_graduable"
        }
    }

    struct Normalizacion: Decodable {
        let prefijoCodigo, prefijoPrograma: String
        let prefijoProgramaSinMayusculas: Bool
        let ruido: [String]
        let sinonimos: [[String]]
        enum CodingKeys: String, CodingKey {
            case prefijoCodigo = "prefijo_codigo", prefijoPrograma = "prefijo_programa",
                 prefijoProgramaSinMayusculas = "prefijo_programa_sin_mayusculas", ruido, sinonimos
        }
    }

    struct ConfigTfidf: Decodable {
        let vocabulario: [String]
        let idf: [Double]
        let ngramas: [Int]
        let quitarStopWords, tfSublineal: Bool
        let patron: String
        enum CodingKeys: String, CodingKey {
            case vocabulario, idf, ngramas, quitarStopWords = "quitar_stop_words", tfSublineal = "tf_sublineal", patron
        }
    }

    struct Categorica: Decodable {
        let columna: String
        let categorias: [String]
    }

    struct Numericas: Decodable {
        let columnas: [String]
        let escala: [Double]
    }

    struct Preprocesador: Decodable {
        let nombre, instrucciones: ConfigTfidf
        let categoricas: [Categorica]
        let numericas: Numericas
        let nVariables: Int
        enum CodingKeys: String, CodingKey {
            case nombre, instrucciones, categoricas, numericas, nVariables = "n_variables"
        }
    }

    struct Logistica: Decodable {
        let coeficientes: [[Double]]
        let intercepto: [Double]
    }

    struct Arbol: Decodable {
        let izq, der, variable: [Int]
        let umbral: [Double]
        let valor: [[Double]]
    }

    struct Modelo: Decodable {
        let algoritmo: String
        let clases: [String]
        let umbralAbstencion: Double
        let preprocesador: Preprocesador
        let logistica: Logistica?
        let bosque: [Arbol]?
        enum CodingKeys: String, CodingKey {
            case algoritmo, clases, umbralAbstencion = "umbral_abstencion", preprocesador, logistica, bosque
        }
    }

    struct TablaPerfil: Decodable {
        let columnas: [String]
        let filas: [String: [ValorJSON]]
    }

    public let ontologia: Ontologia
    let reglas: Reglas
    let normalizacion: Normalizacion
    let stopWords: [String]
    let modelos: [String: Modelo]
    let perfil: TablaPerfil
    /// Nombre en español de los 68 ejercicios curados de la web, por `ejercicio_id`.
    public let nombresEs: [String: String]

    enum CodingKeys: String, CodingKey {
        case ontologia, reglas, normalizacion, stopWords = "stop_words", modelos, perfil, nombresEs = "nombres_es"
    }

    public static func cargar(_ datos: Data) throws -> ModeloPerfil {
        // Sin `convertFromSnakeCase`: también cambiaría las claves de los diccionarios
        // (`free_exercise_db:…`, `complejidad_tecnica`). Cada tipo declara sus claves.
        try JSONDecoder().decode(ModeloPerfil.self, from: datos)
    }
}

/// Un valor JSON de tipo mixto (texto, número o nulo), como las filas del perfil.
enum ValorJSON: Decodable {
    case texto(String), numero(Double), nulo

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .nulo }
        else if let n = try? c.decode(Double.self) { self = .numero(n) }
        else { self = .texto(try c.decode(String.self)) }
    }

    var texto: String? { if case .texto(let t) = self { t } else { nil } }
    var numero: Double? { if case .numero(let n) = self { n } else { nil } }
}
