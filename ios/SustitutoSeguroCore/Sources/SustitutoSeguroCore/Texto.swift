import Foundation

/// Expresión regular compatible con la sintaxis de Python que usa el pipeline.
///
/// `NSRegularExpression` (ICU) entiende `\b`, `\w`, `\s` y `(?:…)` igual que
/// `re` de Python con texto Unicode. El prefijo `(?u)` de scikit-learn no existe
/// en ICU (allí Unicode es el comportamiento por defecto), así que se quita.
struct Patron {
    let regex: NSRegularExpression

    init(_ patron: String, sinMayusculas: Bool = false) {
        let limpio = patron.hasPrefix("(?u)") ? String(patron.dropFirst(4)) : patron
        // Los patrones vienen del pipeline y los cubren los tests: uno inválido es un error de programación.
        regex = try! NSRegularExpression(pattern: limpio, options: sinMayusculas ? [.caseInsensitive] : [])
    }

    /// Equivale a `re.search(...) is not None`.
    func aparece(en texto: String) -> Bool {
        regex.firstMatch(in: texto, range: NSRange(texto.startIndex..., in: texto)) != nil
    }

    /// Equivale a `re.findall(...)` sin grupos.
    func coincidencias(en texto: String) -> [String] {
        regex.matches(in: texto, range: NSRange(texto.startIndex..., in: texto)).compactMap {
            Range($0.range, in: texto).map { String(texto[$0]) }
        }
    }

    /// Equivale a `re.sub(patron, reemplazo, texto)` con un reemplazo literal.
    func reemplazar(en texto: String, por reemplazo: String) -> String {
        regex.stringByReplacingMatches(in: texto, range: NSRange(texto.startIndex..., in: texto),
                                       withTemplate: NSRegularExpression.escapedTemplate(for: reemplazo))
    }
}

/// Normalización de nombres de `data_enrichment.normalizar_nombre`: quita prefijos
/// de programa, marcas de variante y puntuación, unifica sinónimos y reduce
/// plurales simples.
struct NormalizadorNombres {
    private let prefijoCodigo: Patron
    private let prefijoPrograma: Patron
    private let ruido: [Patron]
    private let sinonimos: [(Patron, String)]
    private let noAlfanumerico = Patron("[^a-z0-9 ]+")

    init(_ config: ModeloPerfil.Normalizacion) {
        prefijoCodigo = Patron(config.prefijoCodigo)
        prefijoPrograma = Patron(config.prefijoPrograma, sinMayusculas: config.prefijoProgramaSinMayusculas)
        ruido = config.ruido.map { Patron($0) }
        sinonimos = config.sinonimos.map { (Patron($0[0]), $0[1]) }
    }

    func normalizar(_ nombre: String) -> String {
        var texto = prefijoCodigo.reemplazar(en: nombre.trimmingCharacters(in: .whitespacesAndNewlines), por: "")
        texto = prefijoPrograma.reemplazar(en: texto, por: "").lowercased()
        for patron in ruido { texto = patron.reemplazar(en: texto, por: " ") }
        for (patron, reemplazo) in sinonimos { texto = patron.reemplazar(en: texto, por: reemplazo) }
        texto = noAlfanumerico.reemplazar(en: texto, por: " ")
        let tokens = texto.split(whereSeparator: { $0.isWhitespace }).map { token -> String in
            let t = String(token)
            return t.count > 3 && t.hasSuffix("s") && !t.hasSuffix("ss") ? String(t.dropLast()) : t
        }
        return tokens.joined(separator: " ")
    }
}

/// `TfidfVectorizer` ya ajustado: minúsculas, `token_pattern`, palabras vacías,
/// n-gramas, TF sublineal (1 + ln tf), IDF y norma L2.
struct Tfidf {
    private let indice: [String: Int]
    private let idf: [Double]
    private let ngramas: ClosedRange<Int>
    private let tfSublineal: Bool
    private let token: Patron
    private let stopWords: Set<String>
    var dimension: Int { idf.count }

    init(_ config: ModeloPerfil.ConfigTfidf, stopWords: Set<String>) {
        indice = Dictionary(uniqueKeysWithValues: config.vocabulario.enumerated().map { ($1, $0) })
        idf = config.idf
        ngramas = config.ngramas[0]...config.ngramas[1]
        tfSublineal = config.tfSublineal
        token = Patron(config.patron)
        self.stopWords = config.quitarStopWords ? stopWords : []
    }

    /// Vector disperso: índice de la variable → valor.
    func transformar(_ texto: String) -> [Int: Double] {
        let tokens = token.coincidencias(en: texto.lowercased()).filter { !stopWords.contains($0) }
        var cuentas: [Int: Double] = [:]
        for n in ngramas where n <= tokens.count {
            for inicio in 0...(tokens.count - n) {
                let termino = tokens[inicio..<(inicio + n)].joined(separator: " ")
                if let j = indice[termino] { cuentas[j, default: 0] += 1 }
            }
        }
        var vector = cuentas.mapValues { tfSublineal ? 1 + log($0) : $0 }
        for (j, v) in vector { vector[j] = v * idf[j] }
        let norma = sqrt(vector.values.reduce(0) { $0 + $1 * $1 })
        if norma > 0 { for (j, v) in vector { vector[j] = v / norma } }
        return vector
    }
}
