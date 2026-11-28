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

/// Todo lo que la persona registra en la app, en `Documentos/datos_app.json`. Sin cuenta vive
/// solo en el teléfono; con cuenta, el perfil, la rutina, las sesiones, el peso y los aportes se
/// copian también al servidor (ver `Cuenta`).
@Observable
final class Almacen {
    struct Contenido: Codable {
        var perfil: Perfil?
        var rutina: Rutina?
        var entrenamientos: [Sesion] = []          // sesiones de la rutina (formato de la API)
        var pesos: [Pesaje] = []
        var enCurso: EnCurso?
        var revisiones: [RevisionRecomendacion] = []
        var correcciones: [CorreccionEtiqueta] = []
        var ejerciciosNuevos: [EjercicioNuevo] = []
        var sesiones: [SesionEntrenamiento] = []    // registro libre de versiones anteriores de la app

        init() {}

        private enum CodingKeys: String, CodingKey {
            case perfil, rutina, entrenamientos, pesos, enCurso, revisiones, correcciones, ejerciciosNuevos, sesiones
        }

        /// Cada campo es opcional: así un archivo de una versión anterior de la app se sigue leyendo.
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            perfil = try? c.decodeIfPresent(Perfil.self, forKey: .perfil)
            rutina = try? c.decodeIfPresent(Rutina.self, forKey: .rutina)
            entrenamientos = (try? c.decodeIfPresent([Sesion].self, forKey: .entrenamientos)) ?? []
            pesos = (try? c.decodeIfPresent([Pesaje].self, forKey: .pesos)) ?? []
            enCurso = try? c.decodeIfPresent(EnCurso.self, forKey: .enCurso)
            revisiones = (try? c.decodeIfPresent([RevisionRecomendacion].self, forKey: .revisiones)) ?? []
            correcciones = (try? c.decodeIfPresent([CorreccionEtiqueta].self, forKey: .correcciones)) ?? []
            ejerciciosNuevos = (try? c.decodeIfPresent([EjercicioNuevo].self, forKey: .ejerciciosNuevos)) ?? []
            sesiones = (try? c.decodeIfPresent([SesionEntrenamiento].self, forKey: .sesiones)) ?? []
        }
    }

    var contenido: Contenido { didSet { guardar(); alCambiar?() } }
    private(set) var errorAlGuardar: String?
    /// La cuenta se suscribe para subir los cambios.
    @ObservationIgnored var alCambiar: (() -> Void)?
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

    // MARK: Rutina y entrenamiento

    /// Guarda el perfil y arma la rutina nueva (o una variante de la actual).
    func guardarPerfil(_ perfil: Perfil, generador: GeneradorRutina) {
        let p = perfil.normalizado()
        var c = contenido
        c.perfil = p
        c.rutina = generador.generar(p, hoy: Fechas.iso())
        c.enCurso = nil
        contenido = c
    }

    func otraVariante(generador: GeneradorRutina) {
        guard let p = contenido.perfil else { return }
        var azar = SystemRandomNumberGenerator()
        contenido.rutina = GeneradorRutina.variante(de: generador.generar(p, hoy: Fechas.iso()), generador: &azar)
    }

    func cambiarEjercicio(dia: Int, posicion: Int) {
        guard var r = contenido.rutina else { return }
        GeneradorRutina.cambiar(&r, dia: dia, posicion: posicion)
        contenido.rutina = r
    }

    func empezar(dia: Int) {
        guard let r = contenido.rutina, r.dias.indices.contains(dia) else { return }
        contenido.enCurso = EnCurso(rutina: r, dia: dia, sesiones: contenido.entrenamientos)
    }

    func terminarSesion() {
        guard let s = contenido.enCurso else { return }
        var c = contenido
        c.entrenamientos.append(s.terminar(hoy: Fechas.iso()))
        c.entrenamientos.sort { $0.fecha < $1.fecha }
        c.enCurso = nil
        contenido = c
    }

    func registrarPeso(_ kg: Double) {
        let hoy = Fechas.iso()
        var pesos = contenido.pesos.filter { $0.fecha != hoy }
        pesos.append(Pesaje(fecha: hoy, kg: kg))
        contenido.pesos = pesos.sorted { $0.fecha < $1.fecha }
    }

    // MARK: Cuenta

    /// Lo que se guarda en la cuenta, en el formato de la API.
    var datosCuenta: DatosCuenta {
        let c = contenido
        let correcciones = c.correcciones.map {
            CorreccionAporte(ejercicio_id: $0.ejercicioId, objetivo: $0.objetivo, etiqueta: $0.etiquetaCorregida,
                             etiqueta_previa: $0.etiquetaPipeline ?? "", origen_previo: $0.origenPipeline ?? "",
                             confianza_previa: $0.confianzaPipeline, fecha: CSV.fecha($0.fecha))
        }
        let revisiones = c.revisiones.map {
            RevisionAporte(original_id: $0.originalId, alternativa_id: $0.alternativaId, articulacion: $0.articulacion,
                           perfil: $0.principiante ? "principiante" : "cualquiera", veredicto: $0.veredicto.rawValue,
                           similitud: $0.similitud, misma_familia: $0.mismaFamilia, fecha: CSV.fecha($0.fecha))
        }
        return DatosCuenta(perfil: c.perfil, rutina: c.rutina, sesiones: c.entrenamientos, pesos: c.pesos,
                           correcciones: correcciones, revisiones: revisiones)
    }

    /// Reemplaza lo local con lo de la cuenta (los aportes solo se suben: quedan como están).
    func aplicar(_ d: DatosCuenta) {
        var c = contenido
        c.perfil = d.perfil
        c.rutina = d.rutina
        c.entrenamientos = d.sesiones.sorted { $0.fecha < $1.fecha }
        c.pesos = d.pesos.sorted { $0.fecha < $1.fecha }
        if let enCurso = c.enCurso, !(d.rutina?.dias.indices.contains(enCurso.dia) ?? false) { c.enCurso = nil }
        contenido = c
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
            + contenido.sesiones.count + contenido.entrenamientos.count + contenido.pesos.count
            + (contenido.perfil == nil ? 0 : 1)
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

    /// Nombres de los ejercicios para la exportación (los pone la vista, que tiene el catálogo).
    @ObservationIgnored var nombres: [String: String] = [:]

    // MARK: Exportación

    /// Escribe los cinco CSV en una carpeta temporal y devuelve sus rutas, para
    /// compartirlos (AirDrop, Archivos, correo) y pasarlos al proyecto con
    /// `python scripts/importar_aportes.py <carpeta>`.
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
            let libres = c.sesiones.flatMap { s in
                s.series.enumerated().map { k, serie in
                    [CSV.fecha(s.fecha), s.id.uuidString, serie.ejercicioId, serie.nombre, String(k + 1),
                     CSV.numero(serie.pesoKg, decimales: 1), String(serie.repeticiones), String(serie.esfuerzo)]
                }
            }
            let deRutina = c.entrenamientos.flatMap { s in
                s.ejercicios.flatMap { e in
                    e.series.enumerated().map { k, serie in
                        [s.fecha, s.uid, e.id, nombres[e.id] ?? e.id, String(k + 1),
                         CSV.numero(serie.kg ?? 0, decimales: 1), String(serie.reps), String(s.esfuerzo)]
                    }
                }
            }
            return libres + deRutina
        case .dolor:
            return c.sesiones.flatMap { s in
                s.dolores.map { [CSV.fecha(s.fecha), s.id.uuidString, $0.zona, String($0.dolor)] }
            } + c.entrenamientos.flatMap { s in
                s.dolor.sorted { $0.key < $1.key }.map { [s.fecha, s.uid, TablasRutina.nombreLesion($0.key).lowercased(), String($0.value)] }
            }
        }
    }
}
