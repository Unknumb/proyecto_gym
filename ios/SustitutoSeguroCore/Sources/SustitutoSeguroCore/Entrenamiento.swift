import Foundation

// Sesiones de entrenamiento, peso corporal y estadísticas de progreso: las mismas reglas de
// la página (web/app.html, «ENTRENAR» y «PROGRESO») y el mismo formato que la API de cuentas.

// MARK: - Fechas como en la web: texto AAAA-MM-DD, semanas que empiezan el lunes

public enum Fechas {
    private static let calendario: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        c.firstWeekday = 2
        return c
    }()

    private static func formato(_ zona: TimeZone) -> DateFormatter {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = zona
        f.dateFormat = "yyyy-MM-dd"
        return f
    }

    /// La fecha de hoy (o de `fecha`) en la zona horaria del teléfono.
    public static func iso(_ fecha: Date = Date(), zona: TimeZone = .current) -> String {
        formato(zona).string(from: fecha)
    }

    static func fecha(_ iso: String) -> Date? { formato(TimeZone(identifier: "UTC")!).date(from: iso) }

    public static func esValida(_ iso: String) -> Bool { fecha(iso) != nil && iso.count == 10 }

    /// El lunes de la semana de `iso`.
    public static func lunes(_ iso: String) -> String {
        guard let d = fecha(iso) else { return iso }
        let k = (calendario.component(.weekday, from: d) + 5) % 7     // lunes = 0
        return Self.iso(calendario.date(byAdding: .day, value: -k, to: d)!, zona: TimeZone(identifier: "UTC")!)
    }

    /// Los lunes de las últimas `n` semanas, de la más antigua a la actual.
    public static func semanas(_ n: Int, hoy: String) -> [String] {
        guard let l = fecha(lunes(hoy)) else { return [] }
        return (0..<n).reversed().map {
            iso(calendario.date(byAdding: .day, value: -7 * $0, to: l)!, zona: TimeZone(identifier: "UTC")!)
        }
    }

    /// «12/3»: día y mes, para los ejes de los gráficos.
    public static func corta(_ iso: String) -> String {
        guard let d = fecha(iso) else { return iso }
        let c = calendario.dateComponents([.day, .month], from: d)
        return "\(c.day!)/\(c.month!)"
    }

    /// «mié 12 mar», para el historial.
    public static func larga(_ iso: String) -> String {
        guard let d = fecha(iso) else { return iso }
        let f = DateFormatter()
        f.locale = Locale(identifier: "es_CL")
        f.timeZone = TimeZone(identifier: "UTC")
        f.setLocalizedDateFormatFromTemplate("EEE d MMM")
        return f.string(from: d)
    }
}

// MARK: - Lo que se guarda (formato de la API)

public struct SerieHecha: Codable, Hashable, Sendable {
    public var kg: Double?
    public var reps: Int
    public init(kg: Double?, reps: Int) { self.kg = kg; self.reps = reps }
}

public struct EjercicioHecho: Codable, Hashable, Sendable {
    public var id: String
    public var series: [SerieHecha]
    public init(id: String, series: [SerieHecha]) { self.id = id; self.series = series }
}

/// Una sesión terminada. `uid` la identifica en el teléfono, en la web y en la base de datos.
public struct Sesion: Codable, Hashable, Identifiable, Sendable {
    public var uid: String
    public var fecha: String
    public var dia: Int
    public var titulo: String
    public var minutos: Int
    public var esfuerzo: Int
    public var notas: String
    public var dolor: [String: Int]          // clave de lesión → 0 a 10
    public var ejercicios: [EjercicioHecho]
    public var origen: String

    public var id: String { uid }

    public init(uid: String = Sesion.nuevoUid(), fecha: String, dia: Int, titulo: String, minutos: Int, esfuerzo: Int,
                notas: String = "", dolor: [String: Int] = [:], ejercicios: [EjercicioHecho], origen: String = "ios") {
        self.uid = uid; self.fecha = fecha; self.dia = dia; self.titulo = titulo; self.minutos = minutos
        self.esfuerzo = esfuerzo; self.notas = notas; self.dolor = dolor; self.ejercicios = ejercicios; self.origen = origen
    }

    public static func nuevoUid() -> String { UUID().uuidString.lowercased() }

    public var series: Int { ejercicios.reduce(0) { $0 + $1.series.count } }
}

public struct Pesaje: Codable, Hashable, Identifiable, Sendable {
    public var fecha: String
    public var kg: Double
    public var id: String { fecha }
    public init(fecha: String, kg: Double) { self.fecha = fecha; self.kg = kg }
}

// MARK: - La sesión en curso

public struct SerieEnCurso: Codable, Hashable, Sendable {
    public var kg: Double?
    public var reps: Int
    public var hecho: Bool
}

public struct EjercicioEnCurso: Codable, Hashable, Sendable {
    public var id: String
    public var iso: Bool
    public var descanso: Int
    public var objetivo: String          // «3 × 10–12», como lo pide la rutina
    public var series: [SerieEnCurso]
}

public struct EnCurso: Codable, Hashable, Sendable {
    public var dia: Int
    public var titulo: String
    public var inicio: Date
    public var ejercicios: [EjercicioEnCurso]
    public var esfuerzo = 6
    public var dolor: [String: Int] = [:]
    public var notas = ""

    /// Arma la sesión del día `dia` de la rutina. Los kilos parten de la última vez que se hizo
    /// cada ejercicio, serie por serie, como en la web.
    public init(rutina: Rutina, dia: Int, sesiones: [Sesion], inicio: Date = Date()) {
        let d = rutina.dias[dia]
        self.dia = dia
        self.titulo = d.titulo
        self.inicio = inicio
        ejercicios = d.ejercicios.map { e in
            let previo = sesiones.reversed().lazy.compactMap { s in s.ejercicios.first { $0.id == e.id } }.first
            let reps = Int(e.reps.prefix { $0.isNumber }) ?? 10
            return EjercicioEnCurso(id: e.id, iso: e.iso, descanso: e.descanso, objetivo: "\(e.series) × \(e.reps)",
                                    series: (0..<e.series).map { k in
                                        let kg = previo.flatMap { p in (k < p.series.count ? p.series[k].kg : nil) ?? p.series.first?.kg }
                                        return SerieEnCurso(kg: e.iso ? nil : kg, reps: e.iso ? 30 : reps, hecho: false)
                                    })
        }
        dolor = Dictionary(uniqueKeysWithValues: (rutina.perfil?.lesiones ?? []).map { ($0, 0) })
    }

    public var totalSeries: Int { ejercicios.reduce(0) { $0 + $1.series.count } }
    public var seriesHechas: Int { ejercicios.reduce(0) { $0 + $1.series.filter(\.hecho).count } }

    /// La sesión que se guarda: solo las series marcadas como hechas.
    public func terminar(hoy: String, ahora: Date = Date(), uid: String = Sesion.nuevoUid()) -> Sesion {
        let minutos = min(300, max(1, Int((ahora.timeIntervalSince(inicio) / 60).rounded())))
        let hechos = ejercicios.map { e in
            EjercicioHecho(id: e.id, series: e.series.filter(\.hecho).map { SerieHecha(kg: e.iso ? nil : $0.kg, reps: $0.reps) })
        }.filter { !$0.series.isEmpty }
        return Sesion(uid: uid, fecha: hoy, dia: dia, titulo: titulo, minutos: minutos, esfuerzo: min(10, max(1, esfuerzo)),
                      notas: String(notas.prefix(500)), dolor: dolor.mapValues { min(10, max(0, $0)) }, ejercicios: hechos)
    }

    /// Aviso según el dolor máximo que se marcó al terminar.
    public var alertaDolor: Aviso? { Progreso.alertaAlTerminar(dolor) }
}

// MARK: - Progreso

public enum Progreso {
    public struct Resumen: Hashable, Sendable {
        public var total: Int
        public var estaSemana: Int
        public var racha: Int          // semanas seguidas con al menos una sesión
        public var objetivo: Int       // sesiones por semana (días del perfil)
    }

    public static func resumen(_ sesiones: [Sesion], hoy: String, objetivo: Int) -> Resumen {
        let semana = Fechas.lunes(hoy)
        let porSemana = Set(sesiones.map { Fechas.lunes($0.fecha) })
        var racha = 0
        for (k, w) in Fechas.semanas(104, hoy: hoy).reversed().enumerated() {
            if porSemana.contains(w) { racha += 1 }
            else if k == 0 { continue }       // la semana actual aún puede completarse
            else { break }
        }
        return Resumen(total: sesiones.count, estaSemana: sesiones.filter { Fechas.lunes($0.fecha) == semana }.count,
                       racha: racha, objetivo: objetivo)
    }

    /// Sesiones por semana en las últimas `n` semanas.
    public static func porSemana(_ sesiones: [Sesion], hoy: String, n: Int = 8) -> [(lunes: String, sesiones: Int)] {
        Fechas.semanas(n, hoy: hoy).map { w in (w, sesiones.filter { Fechas.lunes($0.fecha) == w }.count) }
    }

    /// El día que toca: el siguiente al de la última sesión.
    public static func diaSugerido(_ sesiones: [Sesion], dias: Int) -> Int {
        guard dias > 0, let ultima = sesiones.last else { return 0 }
        return (ultima.dia + 1) % dias
    }

    /// Avisos de tendencia del dolor por zona: alto en la última sesión, subiendo o bajando
    /// respecto del promedio de las tres anteriores.
    public static func alertas(_ sesiones: [Sesion]) -> [Aviso] {
        var avisos: [Aviso] = []
        for clave in zonasConDolor(sesiones) {
            let v = sesiones.compactMap { $0.dolor[clave] }
            guard let ultimo = v.last else { continue }
            let previos = v.dropLast().suffix(3)
            let promedio = previos.isEmpty ? nil : Double(previos.reduce(0, +)) / Double(previos.count)
            let nombre = TablasRutina.nombreLesion(clave)
            if ultimo >= 7 {
                avisos.append(Aviso("red", "\(nombre): dolor alto (\(ultimo)/10) en la última sesión.", "Pausa los ejercicios de esa zona y consulta a un profesional."))
            } else if let p = promedio, Double(ultimo) - p >= 2 {
                avisos.append(Aviso("amber", "\(nombre): el dolor está subiendo.", "Baja la carga o cambia los ejercicios que lo provocan. Si sigue subiendo, consulta."))
            } else if let p = promedio, p - Double(ultimo) >= 2 {
                avisos.append(Aviso("teal", "\(nombre): el dolor va bajando.", "Buena señal. Sigue progresando de a poco."))
            }
        }
        return avisos
    }

    public static func alertaAlTerminar(_ dolor: [String: Int]) -> Aviso? {
        let maximo = dolor.values.max() ?? 0
        if maximo >= 7 {
            return Aviso("red", "Dolor alto.", "No sigas entrenando esa zona y consulta a un kinesiólogo o médico antes de la próxima sesión.")
        }
        if maximo >= 4 {
            return Aviso("amber", "Dolor moderado.", "En la próxima sesión baja el peso o cambia el ejercicio que lo provocó. Si persiste, consulta.")
        }
        return nil
    }

    /// Zonas con dolor registrado, en el orden de la lista de lesiones.
    public static func zonasConDolor(_ sesiones: [Sesion]) -> [String] {
        let usadas = Set(sesiones.flatMap { $0.dolor.keys })
        return TablasRutina.lesiones.map(\.clave).filter { usadas.contains($0) }
    }

    /// Ejercicios con al menos una serie con kilos, en el orden en que aparecen.
    public static func ejerciciosConCarga(_ sesiones: [Sesion]) -> [String] {
        var vistos: [String] = []
        for s in sesiones {
            for e in s.ejercicios where e.series.contains(where: { ($0.kg ?? 0) > 0 }) && !vistos.contains(e.id) {
                vistos.append(e.id)
            }
        }
        return vistos
    }

    /// Kilos de la serie más pesada de `id` en cada sesión (las últimas 12).
    public static func cargas(_ sesiones: [Sesion], id: String) -> [(fecha: String, kg: Double)] {
        Array(sesiones.compactMap { s -> (String, Double)? in
            guard let e = s.ejercicios.first(where: { $0.id == id }), e.series.contains(where: { ($0.kg ?? 0) > 0 }) else { return nil }
            return (s.fecha, e.series.map { $0.kg ?? 0 }.max() ?? 0)
        }.suffix(12)).map { (fecha: $0.0, kg: $0.1) }
    }

    public static func esfuerzoPromedio(_ sesiones: [Sesion]) -> Double? {
        sesiones.isEmpty ? nil : Double(sesiones.reduce(0) { $0 + $1.esfuerzo }) / Double(sesiones.count)
    }
}
