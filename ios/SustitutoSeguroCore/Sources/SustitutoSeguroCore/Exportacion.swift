import Foundation

/// Veredicto de una revisión de recomendación: los mismos cuatro valores que la
/// revisión manual del equipo (`data/01_raw/sustitucion_revision_manual.csv`).
public enum Veredicto: String, CaseIterable, Codable, Identifiable {
    case bueno, aceptable, malo, inseguro
    public var id: String { rawValue }
    public var titulo: String {
        switch self {
        case .bueno: "Buena"
        case .aceptable: "Aceptable"
        case .malo: "Mala"
        case .inseguro: "Insegura"
        }
    }
}

/// Tablas que la app exporta para el pipeline. Cada una es un CSV con encabezado
/// que `scripts/importar_datos_ios.py` valida y copia a `data/01_raw/app_ios/`.
public enum TablaExportable: String, CaseIterable {
    case revisiones = "revisiones_recomendaciones"
    case correcciones = "correcciones_etiquetas"
    case ejerciciosNuevos = "ejercicios_nuevos"
    case series = "entrenamiento_series"
    case dolor = "entrenamiento_dolor"

    public var archivo: String { "\(rawValue).csv" }

    /// Columnas, en orden. Las nueve primeras de `revisiones` son las de la revisión
    /// manual del equipo, para poder sumarlas sin transformar.
    public var columnas: [String] {
        switch self {
        case .revisiones:
            ["articulacion_vetada", "original", "familia_original", "alternativa", "familia_alternativa",
             "misma_familia", "similitud", "veredicto", "comentario",
             "original_id", "alternativa_id", "perfil", "fecha"]
        case .correcciones:
            ["ejercicio_id", "nombre", "objetivo", "etiqueta_pipeline", "origen_pipeline", "confianza_pipeline",
             "etiqueta_corregida", "comentario", "fecha"]
        case .ejerciciosNuevos:
            ["nombre", "nombre_norm", "instrucciones_en", "familia_equipamiento", "musculos_primarios",
             "musculos_secundarios", "mecanica_modelo", "mecanica_confianza", "mecanica",
             "fuerza_modelo", "fuerza_confianza", "fuerza",
             "complejidad_tecnica_modelo", "complejidad_tecnica_confianza", "complejidad_tecnica",
             "impacto", "estabilidad", "carga_graduable", "fecha"]
        case .series:
            ["fecha", "sesion", "ejercicio_id", "nombre", "serie", "peso_kg", "repeticiones", "esfuerzo_1_10"]
        case .dolor:
            ["fecha", "sesion", "zona", "dolor_0_10"]
        }
    }
}

public enum CSV {
    /// Un CSV RFC 4180: comillas solo donde hacen falta, comillas internas
    /// duplicadas, saltos de línea `\n` y UTF-8 (lo lee `pandas.read_csv` tal cual).
    public static func escribir(columnas: [String], filas: [[String]]) -> String {
        ([columnas] + filas).map { $0.map(campo).joined(separator: ",") }.joined(separator: "\n") + "\n"
    }

    static func campo(_ valor: String) -> String {
        guard valor.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" || $0 == "\r" }) else { return valor }
        return "\"" + valor.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    /// Fecha ISO 8601 (UTC), la que `pandas.to_datetime` entiende sin formato.
    public static func fecha(_ fecha: Date) -> String { ISO8601DateFormatter().string(from: fecha) }

    public static func numero(_ valor: Double?, decimales: Int = 3) -> String {
        guard let valor else { return "" }
        return String(format: "%.\(decimales)f", locale: Locale(identifier: "en_US_POSIX"), valor)
    }

    /// Listas (músculos) separadas por `;`, como las lee el importador.
    public static func lista(_ valores: [String]) -> String { valores.joined(separator: ";") }
}
