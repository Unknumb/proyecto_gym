// Verifica que la app predice igual que el pipeline, sin Xcode:
//   swift run verificar
// Usa los mismos casos que `SustitutoSeguroCoreTests` (scripts/exportar_modelo_ios.py).
import Foundation
import SustitutoSeguroCore

struct Casos: Decodable {
    struct Caso: Decodable {
        let ejercicio_id, nombre, instrucciones_en, familia_equipamiento: String
        let musculos_primarios, musculos_secundarios: [String]
        let esperado: [String: [Double]]
    }
    struct Nombre: Decodable { let nombre, nombre_norm: String }
    let casos: [Caso]
    let nombres: [Nombre]
}

let raiz = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent()
let inicio = Date()
let perfilador = try Perfilador(datos: Data(contentsOf: raiz.deletingLastPathComponent()
    .appendingPathComponent("SustitutoSeguro/Recursos/modelo_perfil.json")))
let carga = Date().timeIntervalSince(inicio)
let casos = try JSONDecoder().decode(Casos.self, from: Data(contentsOf: raiz
    .appendingPathComponent("Tests/SustitutoSeguroCoreTests/Recursos/casos_prueba.json")))

var fallas = 0
let distintos = casos.nombres.filter { perfilador.normalizar($0.nombre) != $0.nombre_norm }
print("Normalización: \(casos.nombres.count - distintos.count) de \(casos.nombres.count) nombres iguales al pipeline")
for d in distintos.prefix(5) { print("  ✗ \(d.nombre) → \(perfilador.normalizar(d.nombre)) (esperado \(d.nombre_norm))") }
fallas += distintos.count

var peor: [String: Double] = [:]
var etiquetasDistintas = 0
let inicioPrediccion = Date()
for caso in casos.casos {
    let perfil = perfilador.perfil(EntradaEjercicio(
        nombre: caso.nombre, instrucciones: caso.instrucciones_en, familiaEquipamiento: caso.familia_equipamiento,
        musculosPrimarios: caso.musculos_primarios, musculosSecundarios: caso.musculos_secundarios))
    for (objetivo, esperado) in caso.esperado {
        let obtenido = perfil.predicciones[objetivo]!.probabilidades
        let diferencia = zip(obtenido, esperado).map { abs($0 - $1) }.max()!
        peor[objetivo] = max(peor[objetivo] ?? 0, diferencia)
        if diferencia >= 1e-9 {
            fallas += 1
            if fallas <= 10 { print("  ✗ \(objetivo) · \(caso.ejercicio_id) \(caso.nombre): \(obtenido) vs \(esperado)") }
        }
        if obtenido.firstIndex(of: obtenido.max()!) != esperado.firstIndex(of: esperado.max()!) { etiquetasDistintas += 1 }
    }
}
let porEjercicio = Date().timeIntervalSince(inicioPrediccion) / Double(casos.casos.count) * 1000
print("Probabilidades: \(casos.casos.count) ejercicios × 3 objetivos; mayor diferencia con scikit-learn:")
for objetivo in Perfilador.objetivos { print("  \(objetivo): \(peor[objetivo] ?? .nan)") }
print("Clases más probables distintas: \(etiquetasDistintas)")
print(String(format: "Carga del modelo: %.2f s · predicción: %.1f ms por ejercicio (Mac)", carga, porEjercicio))
// Recomendaciones precalculadas (web/datos_web.json), con los nombres en español.
let recomendador = try Recomendador(datos: Data(contentsOf: raiz.deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("web/datos_web.json")), nombresEspanol: perfilador.modelo.nombresEs)
func comprobar(_ ok: Bool, _ texto: String) { print(ok ? "  ✓ \(texto)" : "  ✗ \(texto)"); if !ok { fallas += 1 } }
print("Recomendador: \(recomendador.ejercicios.count) ejercicios, \(recomendador.articulaciones.count) articulaciones")
let sentadilla = recomendador.ejercicio(id: "catalogo:0043")!
let rodilla = recomendador.recomendar(sentadilla, articulacion: "rodilla", principiante: false)
comprobar(rodilla.alternativas.contains { $0.ejercicio.nombre.lowercased().contains("hip thrust") },
          "sentadilla con la rodilla vetada → \(rodilla.alternativas.map(\.ejercicio.nombre))")
comprobar(rodilla.alternativas.allSatisfy { !$0.ejercicio.articulaciones.contains("rodilla") },
          "ninguna alternativa carga la rodilla")
var inseguras = 0, consultas = 0
for e in recomendador.ejercicios {
    for a in e.articulaciones {
        for principiante in [false, true] {
            consultas += 1
            inseguras += recomendador.recomendar(e, articulacion: a, principiante: principiante)
                .alternativas.filter { $0.ejercicio.articulaciones.contains(a) }.count
        }
    }
}
comprobar(inseguras == 0, "\(consultas) consultas, \(inseguras) alternativas que cargan la articulación vetada")
comprobar(recomendador.buscar("sentadilla").first?.nombreEspanol?.hasPrefix("Sentadilla") == true,
          "buscar «sentadilla» → \(recomendador.buscar("sentadilla").prefix(3).map(\.nombreVisible))")
comprobar(recomendador.recomendar(sentadilla, articulacion: "codo", principiante: false).motivo == .noCargaLaArticulacion,
          "la sentadilla no carga el codo: no hace falta sustituirla")
let csv = CSV.escribir(columnas: ["a", "b"], filas: [["x, y", "dijo \"hola\""]])
comprobar(csv == "a,b\n\"x, y\",\"dijo \"\"hola\"\"\"\n", "CSV con comas y comillas")

print(fallas == 0 ? "✓ La app predice y recomienda igual que el pipeline" : "✗ \(fallas) diferencias")
exit(fallas == 0 ? 0 : 1)
