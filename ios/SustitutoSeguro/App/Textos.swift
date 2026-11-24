import SwiftUI

/// Nombres en español de los valores del pipeline (que están en inglés o en
/// minúsculas, como los guardan los datos).
enum Textos {
    static let objetivo = ["mecanica": "Mecánica", "fuerza": "Tipo de fuerza",
                           "complejidad_tecnica": "Complejidad técnica"]

    static let clase = ["compound": "Compuesto", "isolation": "Aislado",
                        "push": "Empuje", "pull": "Tracción", "static": "Estático",
                        "baja": "Baja", "alta": "Alta"]

    static let origen = ["fuente": "de la ficha original", "transferida": "de su gemelo en free-exercise-db",
                         "modelo": "predicha por el modelo"]

    static func clase(_ valor: String?) -> String {
        guard let valor else { return "Sin etiqueta" }
        return clase[valor] ?? valor.capitalized
    }

    static func porcentaje(_ valor: Double) -> String {
        valor.formatted(.percent.precision(.fractionLength(0)))
    }

    static func capitalizar(_ texto: String) -> String {
        texto.prefix(1).uppercased() + texto.dropFirst()
    }
}

extension View {
    /// Título en línea en iOS (en macOS, donde solo se verifica que el código
    /// compila, no existe ese modificador).
    func tituloEnLinea() -> some View {
        #if os(iOS)
        navigationBarTitleDisplayMode(.inline)
        #else
        self
        #endif
    }

    func tecladoDecimal() -> some View {
        #if os(iOS)
        keyboardType(.decimalPad)
        #else
        self
        #endif
    }

    func tecladoNumerico() -> some View {
        #if os(iOS)
        keyboardType(.numberPad)
        #else
        self
        #endif
    }

    func sinMayusculaInicial() -> some View {
        #if os(iOS)
        textInputAutocapitalization(.never)
        #else
        self
        #endif
    }
}
