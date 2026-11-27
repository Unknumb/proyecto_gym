import Foundation
import Testing
@testable import SustitutoSeguroCore

struct EntrenamientoTests {
    private static let raiz = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
    private static let generador: GeneradorRutina = {
        let url = raiz.appendingPathComponent("SustitutoSeguro/Recursos/recomendaciones.json")
        return GeneradorRutina(recomendador: try! Recomendador(datos: Data(contentsOf: url)))
    }()

    private func sesion(_ fecha: String, dia: Int = 0, dolor: [String: Int] = [:], kg: Double? = nil) -> Sesion {
        Sesion(fecha: fecha, dia: dia, titulo: "Día", minutos: 40, esfuerzo: 6, dolor: dolor,
               ejercicios: [EjercicioHecho(id: "catalogo:0025", series: [SerieHecha(kg: kg, reps: 10)])])
    }

    @Test func testSemanasEmpiezanElLunes() {
        #expect(Fechas.lunes("2026-11-27") == "2026-11-23")      // viernes
        #expect(Fechas.lunes("2026-11-23") == "2026-11-23")      // lunes
        #expect(Fechas.lunes("2026-11-29") == "2026-11-23")      // domingo
        #expect(Fechas.lunes("2026-03-01") == "2026-02-23")
        #expect(Fechas.semanas(3, hoy: "2026-11-27") == ["2026-11-09", "2026-11-16", "2026-11-23"])
        #expect(Fechas.corta("2026-03-05") == "5/3")
        #expect(Fechas.esValida("2026-11-27") && !Fechas.esValida("27-11-2026"))
    }

    @Test func testRachaYSesionesDeLaSemana() {
        let s = ["2026-11-02", "2026-11-10", "2026-11-17", "2026-11-18"].map { sesion($0) }
        let r = Progreso.resumen(s, hoy: "2026-11-27", objetivo: 3)
        #expect(r.total == 4 && r.estaSemana == 0 && r.objetivo == 3)
        #expect(r.racha == 3)       // la semana actual sin sesiones no corta la racha
        let r2 = Progreso.resumen(s + [sesion("2026-11-25")], hoy: "2026-11-27", objetivo: 3)
        #expect(r2.estaSemana == 1 && r2.racha == 4)
        #expect(Progreso.porSemana(s, hoy: "2026-11-27", n: 4).map(\.sesiones) == [1, 1, 2, 0])
    }

    @Test func testAlertasDeDolor() {
        let alto = Progreso.alertas([sesion("2026-11-01", dolor: ["rodilla": 2]), sesion("2026-11-03", dolor: ["rodilla": 8])])
        #expect(alto.map(\.color) == ["red"])
        let sube = Progreso.alertas([1, 1, 2, 4].enumerated().map { sesion("2026-11-0\($0.offset + 1)", dolor: ["hombro": $0.element]) })
        #expect(sube.map(\.color) == ["amber"])
        let baja = Progreso.alertas([5, 5, 5, 2].enumerated().map { sesion("2026-11-0\($0.offset + 1)", dolor: ["lumbar": $0.element]) })
        #expect(baja.map(\.color) == ["teal"])
        #expect(Progreso.alertaAlTerminar(["rodilla": 7])?.color == "red")
        #expect(Progreso.alertaAlTerminar(["rodilla": 4])?.color == "amber")
        #expect(Progreso.alertaAlTerminar(["rodilla": 3]) == nil)
    }

    @Test func testSesionEnCursoParteDeLaUltimaCarga() {
        let p = Perfil(edad: 30, nivel: "intermedio", objetivo: "musculo", dias: 3, equipo: ["pesas"], lesiones: ["rodilla"])
        let rutina = Self.generador.generar(p, hoy: "2026-11-27")
        let primero = rutina.dias[0].ejercicios[0]
        let previa = Sesion(fecha: "2026-11-20", dia: 0, titulo: "x", minutos: 30, esfuerzo: 6,
                            ejercicios: [EjercicioHecho(id: primero.id, series: [SerieHecha(kg: 20, reps: 10), SerieHecha(kg: 22.5, reps: 8)])])
        var s = EnCurso(rutina: rutina, dia: 0, sesiones: [previa], inicio: Date(timeIntervalSince1970: 0))
        #expect(s.ejercicios.count == rutina.dias[0].ejercicios.count)
        #expect(s.dolor == ["rodilla": 0])
        let e = s.ejercicios[0]
        #expect(e.series.count == primero.series)
        if !e.iso {
            let base: [Double?] = [20, 22.5, 20, 20, 20, 20]
            let esperado = Array(base.prefix(primero.series))
            #expect(e.series.map(\.kg) == esperado)
            #expect(e.series[0].reps == Int(primero.reps.prefix { $0.isNumber }))
        }
        s.ejercicios[0].series[0].hecho = true
        s.dolor["rodilla"] = 12
        s.esfuerzo = 8
        let fin = s.terminar(hoy: "2026-11-27", ahora: Date(timeIntervalSince1970: 45 * 60 + 10), uid: "abc")
        #expect(fin.uid == "abc" && fin.fecha == "2026-11-27" && fin.minutos == 45 && fin.esfuerzo == 8)
        #expect(fin.ejercicios.count == 1 && fin.ejercicios[0].series.count == 1)
        #expect(fin.dolor == ["rodilla": 10])
        #expect(fin.origen == "ios")
    }

    @Test func testCargasPorEjercicio() {
        let s = [sesion("2026-11-01", kg: 20), sesion("2026-11-03"), sesion("2026-11-05", kg: 25)]
        #expect(Progreso.ejerciciosConCarga(s) == ["catalogo:0025"])
        #expect(Progreso.cargas(s, id: "catalogo:0025").map(\.kg) == [20, 25])
        #expect(Progreso.diaSugerido(s, dias: 3) == 1)
    }

    @Test func testSoloSubeLoQueCambio() throws {
        let p = Perfil(edad: 30, dias: 3)
        let a = sesion("2026-11-01"), b = sesion("2026-11-02")
        let servidor = DatosCuenta(perfil: p, sesiones: [a, b], pesos: [Pesaje(fecha: "2026-11-01", kg: 70)])
        let foto = FotoCuenta(servidor)
        #expect(CambiosCuenta.entre(foto, servidor).vacio)

        var local = servidor
        local.perfil?.dias = 4
        local.sesiones = [a, sesion("2026-11-03")]
        local.pesos = [Pesaje(fecha: "2026-11-02", kg: 71)]
        local.correcciones = [CorreccionAporte(ejercicio_id: "catalogo:0025", objetivo: "mecanica", etiqueta: "compound",
                                               etiqueta_previa: "isolation", origen_previo: "modelo", confianza_previa: 0.7, fecha: "2026-11-27")]
        let c = CambiosCuenta.entre(foto, local)
        #expect(c.perfil?.dias == 4)
        #expect(c.sesiones.map(\.fecha) == ["2026-11-03"])
        #expect(c.quitarSesiones == [b.uid])
        #expect(c.pesos.map(\.fecha) == ["2026-11-02"] && c.quitarPesos == ["2026-11-01"])
        #expect(c.correcciones.count == 1 && !c.quitarRutina && c.rutina == nil)

        // Lo que responde el servidor se decodifica igual que lo que se guardó
        let ida = try JSONDecoder().decode(DatosCuenta.self, from: JSONEncoder().encode(local))
        #expect(CambiosCuenta.entre(FotoCuenta(ida), local).vacio)
    }

    @Test func testDireccionDelServidor() {
        #expect(ClienteCuentas.direccion("3.85.12.40.sslip.io/")?.absoluteString == "https://3.85.12.40.sslip.io")
        #expect(ClienteCuentas.direccion(" http://localhost:8000 ")?.absoluteString == "http://localhost:8000")
        #expect(ClienteCuentas.direccion("") == nil)
    }
}
