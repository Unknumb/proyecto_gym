import Foundation

/// Lo que una persona ingresa para un ejercicio nuevo: los mismos campos crudos
/// que el pipeline usa en `construir_tabla_modelado`.
public struct EntradaEjercicio: Equatable {
    public var nombre: String
    public var instrucciones: String
    public var familiaEquipamiento: String
    public var musculosPrimarios: [String]
    public var musculosSecundarios: [String]

    public init(nombre: String, instrucciones: String, familiaEquipamiento: String,
                musculosPrimarios: [String], musculosSecundarios: [String]) {
        self.nombre = nombre
        self.instrucciones = instrucciones
        self.familiaEquipamiento = familiaEquipamiento
        self.musculosPrimarios = musculosPrimarios
        self.musculosSecundarios = musculosSecundarios
    }
}

/// Reglas transparentes del perfil (`derivar_perfil_reglas`).
public struct PerfilReglas: Equatable {
    public let impacto: Bool
    public let estabilidad: String      // "inestable" | "apoyado" | "libre"
    public let cargaGraduable: Bool
    public let esEstiramiento: Bool
}

/// La predicción de un objetivo: probabilidades por clase y la etiqueta, o `nil`
/// si el modelo se abstiene porque su confianza no alcanza el umbral.
public struct Prediccion: Equatable {
    public let clases: [String]
    public let probabilidades: [Double]
    public let umbral: Double
    public var confianza: Double { probabilidades.max() ?? 0 }
    public var claseMasProbable: String { clases[probabilidades.firstIndex(of: confianza) ?? 0] }
    public var etiqueta: String? { confianza >= umbral ? claseMasProbable : nil }
}

/// Perfil de exigencia completo de un ejercicio nuevo.
public struct PerfilEjercicio: Equatable {
    public let nombreNormalizado: String
    public let reglas: PerfilReglas
    public let predicciones: [String: Prediccion]   // "mecanica", "fuerza", "complejidad_tecnica"
}

/// Perfil ya calculado por el pipeline para un ejercicio del grafo.
public struct PerfilGuardado: Equatable {
    public let etiqueta: String?
    public let origen: String?        // "fuente" | "transferida" | "modelo" | nil (abstención)
    public let confianza: Double?
}

/// Predice el perfil de exigencia en el dispositivo, con los modelos exportados de
/// `modelado_supervisado`. Da las mismas probabilidades que scikit-learn
/// (`SustitutoSeguroCoreTests`).
public final class Perfilador {
    public static let objetivos = ["mecanica", "fuerza", "complejidad_tecnica"]

    public let modelo: ModeloPerfil
    private let normalizador: NormalizadorNombres
    private let preparados: [String: ModeloPreparado]
    private let impacto, sinImpacto, inestable, apoyado, estiramiento: Patron

    public init(_ modelo: ModeloPerfil) {
        self.modelo = modelo
        normalizador = NormalizadorNombres(modelo.normalizacion)
        let stopWords = Set(modelo.stopWords)
        preparados = modelo.modelos.mapValues { ModeloPreparado($0, stopWords: stopWords) }
        let r = modelo.reglas
        impacto = Patron(r.impacto)
        sinImpacto = Patron(r.sinImpacto)
        inestable = Patron(r.inestable)
        apoyado = Patron(r.apoyado)
        estiramiento = Patron(r.estiramiento)
    }

    public convenience init(datos: Data) throws {
        self.init(try ModeloPerfil.cargar(datos))
    }

    public func normalizar(_ nombre: String) -> String { normalizador.normalizar(nombre) }

    /// Clases del modelo de un objetivo, en el orden de scikit-learn.
    public func clases(_ objetivo: String) -> [String] { modelo.modelos[objetivo]?.clases ?? [] }

    /// Confianza mínima para etiquetar (umbral de abstención del pipeline).
    public func umbral(_ objetivo: String) -> Double { modelo.modelos[objetivo]?.umbralAbstencion ?? 1 }

    /// Algoritmo ganador de un objetivo (`regresion_logistica` o `random_forest`).
    public func algoritmo(_ objetivo: String) -> String { modelo.modelos[objetivo]?.algoritmo ?? "" }

    public func reglas(nombreNormalizado n: String, familia: String) -> PerfilReglas {
        let r = modelo.reglas
        let esInestable = inestable.aparece(en: n) || r.familiasInestable.contains(familia)
        let esApoyado = apoyado.aparece(en: n) || r.familiasApoyado.contains(familia)
        return PerfilReglas(
            impacto: impacto.aparece(en: n) && !sinImpacto.aparece(en: n),
            estabilidad: esInestable ? "inestable" : (esApoyado ? "apoyado" : "libre"),
            cargaGraduable: r.familiasCargaGraduable.contains(familia),
            esEstiramiento: estiramiento.aparece(en: n))
    }

    public func perfil(_ entrada: EntradaEjercicio) -> PerfilEjercicio {
        let nombre = normalizar(entrada.nombre)
        let reglas = reglas(nombreNormalizado: nombre, familia: entrada.familiaEquipamiento)
        var predicciones: [String: Prediccion] = [:]
        for (objetivo, preparado) in preparados {
            predicciones[objetivo] = preparado.predecir(entrada, nombreNormalizado: nombre, reglas: reglas,
                                                        ontologia: modelo.ontologia)
        }
        return PerfilEjercicio(nombreNormalizado: nombre, reglas: reglas, predicciones: predicciones)
    }

    /// El perfil que el pipeline guardó para un ejercicio del grafo (`perfil_exigencia`).
    public func perfilGuardado(ejercicioId: String, objetivo: String) -> PerfilGuardado? {
        guard let fila = modelo.perfil.filas[ejercicioId],
              let k = modelo.perfil.columnas.firstIndex(of: objetivo) else { return nil }
        return PerfilGuardado(etiqueta: fila[k].texto, origen: fila[k + 1].texto, confianza: fila[k + 2].numero)
    }
}

/// Un modelo exportado, con sus transformadores listos para usar.
private struct ModeloPreparado {
    let modelo: ModeloPerfil.Modelo
    let nombre: Tfidf
    let instrucciones: Tfidf

    init(_ modelo: ModeloPerfil.Modelo, stopWords: Set<String>) {
        self.modelo = modelo
        nombre = Tfidf(modelo.preprocesador.nombre, stopWords: stopWords)
        instrucciones = Tfidf(modelo.preprocesador.instrucciones, stopWords: stopWords)
    }

    /// Mismo orden de variables que el `ColumnTransformer`: TF-IDF del nombre, de
    /// las instrucciones, *one-hot* de equipamiento y estabilidad, y las variables
    /// anatómicas divididas por su `MaxAbsScaler`.
    func variables(_ e: EntradaEjercicio, nombreNormalizado: String, reglas: PerfilReglas,
                   ontologia: ModeloPerfil.Ontologia) -> [Int: Double] {
        let p = modelo.preprocesador
        var x: [Int: Double] = [:]
        var desplazamiento = 0
        for (j, v) in nombre.transformar(nombreNormalizado) { x[desplazamiento + j] = v }
        desplazamiento += nombre.dimension
        for (j, v) in instrucciones.transformar(e.instrucciones) { x[desplazamiento + j] = v }
        desplazamiento += instrucciones.dimension
        for categorica in p.categoricas {
            let valor = categorica.columna == "estabilidad" ? reglas.estabilidad : e.familiaEquipamiento
            if let k = categorica.categorias.firstIndex(of: valor) { x[desplazamiento + k] = 1 }
            desplazamiento += categorica.categorias.count
        }

        let articulaciones = { (musculos: [String]) in
            Set(musculos.flatMap { ontologia.articulacionesPorMusculo[$0] ?? [] })
        }
        let deAgonistas = articulaciones(e.musculosPrimarios)
        let total = deAgonistas.union(articulaciones(e.musculosSecundarios))
        for (k, columna) in p.numericas.columnas.enumerated() {
            let valor: Double
            switch columna {
            case "n_agonistas": valor = Double(e.musculosPrimarios.count)
            case "n_sinergistas": valor = Double(e.musculosSecundarios.count)
            case "n_articulaciones_agonistas": valor = Double(deAgonistas.count)
            case "n_articulaciones_total": valor = Double(total.count)
            case "impacto": valor = reglas.impacto ? 1 : 0
            case "carga_graduable": valor = reglas.cargaGraduable ? 1 : 0
            case "es_estiramiento_nombre": valor = reglas.esEstiramiento ? 1 : 0
            default:
                if let m = columna.quitandoPrefijo("agonista__") { valor = e.musculosPrimarios.contains(m) ? 1 : 0 }
                else if let m = columna.quitandoPrefijo("sinergista__") { valor = e.musculosSecundarios.contains(m) ? 1 : 0 }
                else if let a = columna.quitandoPrefijo("articulacion__") { valor = total.contains(a) ? 1 : 0 }
                else { preconditionFailure("Variable desconocida: \(columna)") }
            }
            if valor != 0 { x[desplazamiento + k] = valor / p.numericas.escala[k] }
        }
        return x
    }

    func predecir(_ e: EntradaEjercicio, nombreNormalizado: String, reglas: PerfilReglas,
                  ontologia: ModeloPerfil.Ontologia) -> Prediccion {
        let x = variables(e, nombreNormalizado: nombreNormalizado, reglas: reglas, ontologia: ontologia)
        let probabilidades: [Double]
        if let logistica = modelo.logistica {
            let z = zip(logistica.coeficientes, logistica.intercepto).map { fila, b in
                x.reduce(b) { $0 + fila[$1.key] * $1.value }
            }
            if z.count == 1 {
                let p = 1 / (1 + exp(-z[0]))
                probabilidades = [1 - p, p]
            } else {
                let maximo = z.max() ?? 0
                let e = z.map { exp($0 - maximo) }
                let suma = e.reduce(0, +)
                probabilidades = e.map { $0 / suma }
            }
        } else if let bosque = modelo.bosque {
            // Como scikit-learn: los árboles comparan las variables en float32 contra
            // umbrales en float64, y las probabilidades se promedian en float64.
            var suma = [Double](repeating: 0, count: modelo.clases.count)
            for arbol in bosque {
                var nodo = 0
                while arbol.izq[nodo] != -1 {
                    let valor = Double(Float(x[arbol.variable[nodo]] ?? 0))
                    nodo = valor <= arbol.umbral[nodo] ? arbol.izq[nodo] : arbol.der[nodo]
                }
                for (k, p) in arbol.valor[nodo].enumerated() { suma[k] += p }
            }
            probabilidades = suma.map { $0 / Double(bosque.count) }
        } else {
            preconditionFailure("Modelo sin coeficientes ni árboles: \(modelo.algoritmo)")
        }
        return Prediccion(clases: modelo.clases, probabilidades: probabilidades, umbral: modelo.umbralAbstencion)
    }
}

private extension String {
    func quitandoPrefijo(_ prefijo: String) -> String? {
        hasPrefix(prefijo) ? String(dropFirst(prefijo.count)) : nil
    }
}
