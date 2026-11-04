import Foundation
import Testing
@testable import SustitutoSeguroCore

/// Casos de `scripts/exportar_modelo_ios.py --casos`: ejercicios reales del catálogo y
/// de wger con la probabilidad que da scikit-learn.
private struct Casos: Decodable {
    struct Caso: Decodable {
        let ejercicio_id, nombre, instrucciones_en, familia_equipamiento: String
        let musculos_primarios, musculos_secundarios: [String]
        let esperado: [String: [Double]]
    }
    struct Nombre: Decodable { let nombre, nombre_norm: String }
    let casos: [Caso]
    let nombres: [Nombre]
}

struct PerfiladorTests {
    private static let raiz = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
    private static let perfilador: Perfilador = {
        let url = raiz.appendingPathComponent("SustitutoSeguro/Recursos/modelo_perfil.json")
        return try! Perfilador(datos: Data(contentsOf: url))
    }()
    private static let casos: Casos = {
        let url = Bundle.module.url(forResource: "casos_prueba", withExtension: "json", subdirectory: "Recursos")!
        return try! JSONDecoder().decode(Casos.self, from: Data(contentsOf: url))
    }()

    @Test func testNormalizaLosNombresIgualQueElPipeline() {
        let distintos = Self.casos.nombres.filter { Self.perfilador.normalizar($0.nombre) != $0.nombre_norm }
        #expect(distintos.isEmpty, "Primeros distintos: \(distintos.prefix(5).map { ($0.nombre, $0.nombre_norm, Self.perfilador.normalizar($0.nombre)) })")
        #expect(Self.casos.nombres.count > 2000)
    }

    @Test func testDaLasMismasProbabilidadesQueScikitLearn() {
        var peor: [String: Double] = [:]
        for caso in Self.casos.casos {
            let entrada = EntradaEjercicio(nombre: caso.nombre, instrucciones: caso.instrucciones_en,
                                           familiaEquipamiento: caso.familia_equipamiento,
                                           musculosPrimarios: caso.musculos_primarios,
                                           musculosSecundarios: caso.musculos_secundarios)
            let perfil = Self.perfilador.perfil(entrada)
            for (objetivo, esperado) in caso.esperado {
                let obtenido = perfil.predicciones[objetivo]!.probabilidades
                let diferencia = zip(obtenido, esperado).map { abs($0 - $1) }.max()!
                peor[objetivo] = max(peor[objetivo] ?? 0, diferencia)
                #expect(diferencia < 1e-9, "\(objetivo) · \(caso.ejercicio_id) \(caso.nombre): \(obtenido) vs \(esperado)")
            }
        }
        print("Mayor diferencia con scikit-learn por objetivo:", peor)
        #expect(Set(peor.keys) == Set(Perfilador.objetivos))
    }

    @Test func testSeAbstieneBajoElUmbral() {
        let p = Prediccion(clases: ["baja", "alta"], probabilidades: [0.3, 0.7], umbral: 0.89)
        #expect(p.etiqueta == nil)
        #expect(p.claseMasProbable == "alta")
        #expect(Prediccion(clases: ["baja", "alta"], probabilidades: [0.05, 0.95], umbral: 0.89).etiqueta == "alta")
    }

    @Test func testReglasDelPerfil() {
        let r = Self.perfilador.reglas(nombreNormalizado: "single leg box jump", familia: "peso corporal")
        #expect(r.impacto)
        #expect(r.estabilidad == "inestable")
        #expect(!Self.perfilador.reglas(nombreNormalizado: "stationary bike run", familia: "maquina de cardio").impacto)
        #expect(Self.perfilador.reglas(nombreNormalizado: "seated cable row", familia: "cable").estabilidad == "apoyado")
        #expect(Self.perfilador.reglas(nombreNormalizado: "band pull apart", familia: "banda elastica").cargaGraduable)
    }

    @Test func testPerfilGuardadoDelPipeline() {
        let p = Self.perfilador.perfilGuardado(ejercicioId: "catalogo:0001", objetivo: "mecanica")
        #expect(p?.etiqueta == "compound")
        #expect(p?.origen == "transferida")
        #expect(Self.perfilador.perfilGuardado(ejercicioId: "no-existe", objetivo: "mecanica") == nil)
    }
}
