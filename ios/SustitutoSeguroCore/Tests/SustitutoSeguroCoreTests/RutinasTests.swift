import Foundation
import Testing
@testable import SustitutoSeguroCore

/// Rutinas que armó la página (`node scripts/casos_rutina_ios.mjs`) para 60 perfiles fijos.
private struct CasosRutina: Decodable {
    struct Esperada: Decodable {
        let objetivo: String
        let tipo: String
        let porqueTipo: String
        let avisos: [Aviso]
        let huecos: [String]
        let dias: [DiaRutina]
    }
    struct Caso: Decodable {
        let perfil: Perfil
        let rutina: Esperada
    }
    let casos: [Caso]
}

struct RutinasTests {
    private static let raiz = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
    private static let recomendador: Recomendador = {
        let url = raiz.appendingPathComponent("SustitutoSeguro/Recursos/recomendaciones.json")
        return try! Recomendador(datos: Data(contentsOf: url))
    }()
    private static let generador = GeneradorRutina(recomendador: recomendador)
    private static let casos: CasosRutina = {
        let url = Bundle.module.url(forResource: "casos_rutina", withExtension: "json", subdirectory: "Recursos")!
        return try! JSONDecoder().decode(CasosRutina.self, from: Data(contentsOf: url))
    }()

    @Test func testArmaLaMismaRutinaQueLaPagina() {
        #expect(Self.casos.casos.count == 60)
        for (k, caso) in Self.casos.casos.enumerated() {
            let r = Self.generador.generar(caso.perfil, hoy: "2026-11-27")
            let e = caso.rutina
            #expect(r.objetivo == e.objetivo, "caso \(k): objetivo")
            #expect(r.tipo == e.tipo, "caso \(k): tipo \(r.tipo ?? "-") vs \(e.tipo)")
            #expect(r.porqueTipo == e.porqueTipo, "caso \(k): porqueTipo")
            #expect(r.avisos == e.avisos, "caso \(k): avisos \(r.avisos) vs \(e.avisos)")
            #expect(r.huecos == e.huecos, "caso \(k): huecos \(r.huecos) vs \(e.huecos)")
            #expect(r.dias.count == e.dias.count, "caso \(k): días \(r.dias.count) vs \(e.dias.count)")
            for (d, (a, b)) in zip(r.dias, e.dias).enumerated() {
                #expect(a.titulo == b.titulo && a.equilibrio == b.equilibrio, "caso \(k) día \(d): \(a.titulo) vs \(b.titulo)")
                #expect(a.ejercicios.map(\.id) == b.ejercicios.map(\.id),
                        "caso \(k) día \(d): \(a.ejercicios.map(\.id)) vs \(b.ejercicios.map(\.id))")
                #expect(a.ejercicios == b.ejercicios, "caso \(k) día \(d): series, reps, descanso o alternativas distintas")
            }
        }
    }

    @Test func testNingunEjercicioCargaUnaLesion() {
        let todas = TablasRutina.lesiones.map(\.clave)
        for lesion in todas {
            for equipo in [[], ["pesas", "gym", "banda", "balon"]] as [[String]] {
                let p = Perfil(edad: 30, nivel: "avanzado", objetivo: "musculo", dias: 5, minutos: 75, equipo: equipo, lesiones: [lesion])
                let r = Self.generador.generar(p, hoy: "2026-11-27")
                let vetadas = TablasRutina.vetadas([lesion])
                for dia in r.dias {
                    for e in dia.ejercicios {
                        for id in [e.id] + e.alternativas {
                            let arts = Self.recomendador.ejercicio(id: id)!.articulaciones
                            let carga = arts.contains { vetadas.contains($0) }
                            #expect(!carga, "\(lesion): \(id) carga \(arts)")
                        }
                    }
                }
            }
        }
    }

    @Test func testSinImpactoDesde60AnosYConMenores() {
        let mayor = Self.generador.generar(Perfil(edad: 70, nivel: "avanzado", dias: 4, equipo: ["pesas", "gym"]), hoy: "2026-11-27")
        let curados = Dictionary(uniqueKeysWithValues: TablasRutina.curados.map { ($0.id, $0) })
        let todosConEquilibrio = mayor.dias.allSatisfy { $0.equilibrio }
        let conImpacto = mayor.dias.flatMap { $0.ejercicios }.contains { curados[$0.id]?.imp == true }
        #expect(todosConEquilibrio)
        #expect(!conImpacto)
        let nino = Self.generador.generar(Perfil(edad: 12, objetivo: "fuerza", dias: 3, equipo: ["banda"]), hoy: "2026-11-27")
        #expect(nino.objetivo == "salud")
        let todosSimples = nino.dias.flatMap { $0.ejercicios }.allSatisfy { (curados[$0.id]?.nivel ?? 1) == 1 }
        #expect(todosSimples)
        #expect(nino.avisos.first?.titulo == "Menores de 14 años:")
    }

    @Test func testCambiarRotaLasAlternativas() {
        var r = Self.generador.generar(Perfil(edad: 30, dias: 3, equipo: ["pesas"]), hoy: "2026-11-27")
        let antes = r.dias[0].ejercicios[0]
        GeneradorRutina.cambiar(&r, dia: 0, posicion: 0)
        let despues = r.dias[0].ejercicios[0]
        #expect(despues.id == antes.alternativas.first)
        #expect(despues.alternativas.last == antes.id)
        #expect(Set(despues.alternativas + [despues.id]) == Set(antes.alternativas + [antes.id]))
    }

    @Test func testLaVarianteNoRepiteEjerciciosEnUnDia() {
        var azar = SystemRandomNumberGenerator()
        let base = Self.generador.generar(Perfil(edad: 30, nivel: "intermedio", dias: 4, equipo: ["pesas", "gym"]), hoy: "2026-11-27")
        for _ in 0..<20 {
            let v = GeneradorRutina.variante(de: base, generador: &azar)
            for dia in v.dias {
                let ids = dia.ejercicios.map(\.id)
                #expect(Set(ids).count == ids.count, "\(dia.titulo): \(ids)")
            }
        }
    }

    @Test func testPerfilAceptaLoQueGuardaLaWeb() throws {
        let json = #"{"edad":"25","peso":"","altura":170,"deporte":"futbol","nivel":"intermedio","objetivo":"musculo","dias":4,"minutos":60,"equipo":["pc","pesas"],"lesiones":["rodilla"]}"#
        let p = try JSONDecoder().decode(Perfil.self, from: Data(json.utf8))
        #expect(p.edad == 25 && p.peso == nil && p.altura == 170 && p.tipo == "auto")
        #expect(p.normalizado().equipo == ["pesas"])
        let otra = try JSONDecoder().decode(Perfil.self, from: JSONEncoder().encode(p))
        #expect(otra == p)
    }
}
