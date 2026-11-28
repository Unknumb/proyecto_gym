import Foundation

// Generador de rutinas: el mismo de la página (web/app.html, «RUTINA: generador»), regla por
// regla, para que la web y el iPhone armen la misma rutina con los mismos datos.
// `RutinasTests` lo comprueba contra 60 rutinas armadas por la web
// (`scripts/casos_rutina_ios.mjs`), y `tests/test_rutina_ios.py` que las tablas coincidan.

// MARK: - Datos del usuario y de la rutina (mismo formato que la API de cuentas, README §20)

/// Lo que la persona cuenta de sí misma para armar la rutina.
public struct Perfil: Codable, Hashable, Sendable {
    public var edad: Int
    public var peso: Double?
    public var altura: Double?
    public var deporte: String
    public var nivel: String
    public var objetivo: String
    public var dias: Int
    public var minutos: Int
    public var tipo: String
    public var equipo: [String]
    public var lesiones: [String]

    public init(edad: Int, peso: Double? = nil, altura: Double? = nil, deporte: String = "ninguno",
                nivel: String = "principiante", objetivo: String = "salud", dias: Int = 3, minutos: Int = 45,
                tipo: String = "auto", equipo: [String] = [], lesiones: [String] = []) {
        self.edad = edad; self.peso = peso; self.altura = altura; self.deporte = deporte; self.nivel = nivel
        self.objetivo = objetivo; self.dias = dias; self.minutos = minutos; self.tipo = tipo
        self.equipo = equipo; self.lesiones = lesiones
    }

    private enum Claves: String, CodingKey {
        case edad, peso, altura, deporte, nivel, objetivo, dias, minutos, tipo, equipo, lesiones
    }

    /// Acepta números escritos como texto y `""` como vacío (así guarda la web lo que no se completó).
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Claves.self)
        func numero(_ k: Claves) -> Double? {
            if let v = try? c.decode(Double.self, forKey: k) { return v }
            if let t = try? c.decode(String.self, forKey: k) { return Double(t.replacingOccurrences(of: ",", with: ".")) }
            return nil
        }
        edad = Int(numero(.edad) ?? 0)
        peso = numero(.peso)
        altura = numero(.altura)
        deporte = (try? c.decode(String.self, forKey: .deporte)) ?? "ninguno"
        nivel = (try? c.decode(String.self, forKey: .nivel)) ?? "principiante"
        objetivo = (try? c.decode(String.self, forKey: .objetivo)) ?? "salud"
        dias = Int(numero(.dias) ?? 3)
        minutos = Int(numero(.minutos) ?? 45)
        tipo = (try? c.decode(String.self, forKey: .tipo)) ?? "auto"
        equipo = (try? c.decode([String].self, forKey: .equipo)) ?? []
        lesiones = (try? c.decode([String].self, forKey: .lesiones)) ?? []
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Claves.self)
        try c.encode(edad, forKey: .edad)
        try c.encode(peso, forKey: .peso)
        try c.encode(altura, forKey: .altura)
        try c.encode(deporte, forKey: .deporte)
        try c.encode(nivel, forKey: .nivel)
        try c.encode(objetivo, forKey: .objetivo)
        try c.encode(dias, forKey: .dias)
        try c.encode(minutos, forKey: .minutos)
        try c.encode(tipo, forKey: .tipo)
        try c.encode(equipo, forKey: .equipo)
        try c.encode(lesiones, forKey: .lesiones)
    }

    /// Las reglas del formulario de la web: el peso corporal siempre está incluido (no se guarda) y
    /// con menos de 14 años solo se usan bandas además del peso corporal.
    public func normalizado() -> Perfil {
        var p = self
        p.equipo = equipo.filter { $0 != "pc" }
        if edad < 14 { p.equipo = p.equipo.filter { $0 == "banda" } }
        return p
    }
}

/// Un aviso de la rutina: color (`amber`, `red` o `teal`), título y texto. En JSON es
/// `[color, título, texto]`, como en la web.
public struct Aviso: Codable, Hashable, Sendable {
    public var color: String
    public var titulo: String
    public var texto: String

    public init(_ color: String, _ titulo: String, _ texto: String) {
        self.color = color; self.titulo = titulo; self.texto = texto
    }

    public init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        color = try c.decode(String.self); titulo = try c.decode(String.self); texto = try c.decode(String.self)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(color); try c.encode(titulo); try c.encode(texto)
    }
}

public struct ItemRutina: Codable, Hashable, Sendable {
    public var id: String               // ejercicio_id
    public var grupo: String
    public var series: Int
    public var reps: String
    public var descanso: Int            // segundos
    public var iso: Bool                // isométrico: se mide en segundos
    public var extra: Bool              // complemento del deporte
    public var alternativas: [String]   // otras opciones seguras para el mismo hueco
}

public struct DiaRutina: Codable, Hashable, Sendable {
    public var titulo: String
    public var equilibrio: Bool
    public var ejercicios: [ItemRutina]
}

public struct Rutina: Codable, Hashable, Sendable {
    public var creada: String           // AAAA-MM-DD
    public var perfil: Perfil?
    public var objetivo: String
    public var tipo: String?
    public var porqueTipo: String
    public var avisos: [Aviso]
    public var huecos: [String]
    public var dias: [DiaRutina]

    private enum Claves: String, CodingKey { case creada, perfil, objetivo, tipo, porqueTipo, avisos, huecos, dias }

    public init(creada: String, perfil: Perfil?, objetivo: String, tipo: String?, porqueTipo: String,
                avisos: [Aviso], huecos: [String], dias: [DiaRutina]) {
        self.creada = creada; self.perfil = perfil; self.objetivo = objetivo; self.tipo = tipo
        self.porqueTipo = porqueTipo; self.avisos = avisos; self.huecos = huecos; self.dias = dias
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Claves.self)
        creada = (try? c.decode(String.self, forKey: .creada)) ?? ""
        perfil = try? c.decode(Perfil.self, forKey: .perfil)
        objetivo = (try? c.decode(String.self, forKey: .objetivo)) ?? "salud"
        tipo = try? c.decode(String.self, forKey: .tipo)
        porqueTipo = (try? c.decode(String.self, forKey: .porqueTipo)) ?? ""
        avisos = (try? c.decode([Aviso].self, forKey: .avisos)) ?? []
        huecos = (try? c.decode([String].self, forKey: .huecos)) ?? []
        dias = try c.decode([DiaRutina].self, forKey: .dias)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Claves.self)
        try c.encode(creada, forKey: .creada)
        try c.encode(perfil, forKey: .perfil)
        try c.encode(objetivo, forKey: .objetivo)
        try c.encode(tipo, forKey: .tipo)
        try c.encode(porqueTipo, forKey: .porqueTipo)
        try c.encode(avisos, forKey: .avisos)
        try c.encode(huecos, forKey: .huecos)
        try c.encode(dias, forKey: .dias)
    }
}

// MARK: - Tablas (las mismas de la web)

public enum TablasRutina {
    /// Lesión tal como la entiende una persona → articulaciones del modelo (criterio conservador).
    public static let lesiones: [(clave: String, nombre: String, articulaciones: [String])] = [
        ("hombro", "Hombro", ["hombro", "escapulotorácica"]),
        ("codo", "Codo", ["codo", "radiocubital"]),
        ("muneca", "Muñeca", ["muñeca", "muñeca/dedos", "radiocubital"]),
        ("cuello", "Cuello", ["columna (cervical)"]),
        ("lumbar", "Zona lumbar", ["columna (lumbar)", "columna (torácico-lumbar)"]),
        ("cadera", "Cadera", ["cadera"]),
        ("rodilla", "Rodilla", ["rodilla"]),
        ("tobillo", "Tobillo o pie", ["tobillo", "pie (intertarsal)"]),
    ]

    public static func nombreLesion(_ clave: String) -> String {
        lesiones.first { $0.clave == clave }?.nombre ?? clave
    }

    public static func vetadas(_ claves: [String]) -> Set<String> {
        Set(claves.flatMap { c in lesiones.first { $0.clave == c }?.articulaciones ?? [] })
    }

    /// [ejercicio_id, nombre en español, grupo, nivel 1-3, categoría de equipo (nil = la del dato), impacto, isométrico]
    public static let curados: [(id: String, es: String, grupo: String, nivel: Int, eq: String?, imp: Bool, iso: Bool)] = [
        ("free_exercise_db:Bodyweight_Squat", "Sentadilla con peso corporal", "rodilla", 1, nil, false, false),
        ("catalogo:1760", "Sentadilla goblet con mancuerna", "rodilla", 1, nil, false, false),
        ("free_exercise_db:Barbell_Squat", "Sentadilla con barra", "rodilla", 2, nil, false, false),
        ("catalogo:0043", "Sentadilla profunda con barra", "rodilla", 3, nil, false, false),
        ("free_exercise_db:Leg_Press", "Prensa de piernas", "rodilla", 1, nil, false, false),
        ("wger:999", "Zancada hacia atrás", "rodilla", 1, nil, false, false),
        ("catalogo:0336", "Zancada con mancuernas", "rodilla", 2, nil, false, false),
        ("catalogo:1460", "Zancada caminando", "rodilla", 2, nil, false, false),
        ("catalogo:0431", "Subida al cajón con mancuernas", "rodilla", 2, nil, false, false),
        ("wger:632", "Sentadilla sumo", "rodilla", 1, nil, false, false),
        ("catalogo:1001", "Sentadilla dividida con banda", "rodilla", 2, nil, false, false),
        ("catalogo:0585", "Extensión de piernas en máquina", "rodilla", 1, nil, false, false),
        ("wger:1408", "Sentadilla isométrica en pared", "rodilla", 1, "pc", false, true),
        ("wger:265", "Puente de glúteo", "cadera", 1, "pc", false, false),
        ("free_exercise_db:Single_Leg_Glute_Bridge", "Puente de glúteo a una pierna", "cadera", 2, nil, false, false),
        ("wger:294", "Hip thrust con barra", "cadera", 2, nil, false, false),
        ("catalogo:1459", "Peso muerto rumano con mancuernas", "cadera", 2, nil, false, false),
        ("catalogo:0085", "Peso muerto rumano con barra", "cadera", 2, nil, false, false),
        ("catalogo:0032", "Peso muerto con barra", "cadera", 3, nil, false, false),
        ("free_exercise_db:Lying_Leg_Curls", "Curl femoral tumbado", "cadera", 1, nil, false, false),
        ("catalogo:0549", "Swing con kettlebell", "cadera", 3, nil, false, false),
        ("wger:1823", "Almeja (glúteo medio)", "cadera", 1, nil, false, false),
        ("free_exercise_db:Superman", "Superman", "cadera", 1, nil, false, false),
        ("catalogo:0662", "Flexiones de brazos", "pecho", 1, nil, false, false),
        ("catalogo:0289", "Press de banca con mancuernas", "pecho", 1, nil, false, false),
        ("catalogo:0025", "Press de banca con barra", "pecho", 2, nil, false, false),
        ("catalogo:0314", "Press inclinado con mancuernas", "pecho", 2, nil, false, false),
        ("catalogo:0577", "Press de pecho en máquina", "pecho", 1, nil, false, false),
        ("free_exercise_db:Cable_Crossover", "Cruce de poleas", "pecho", 2, nil, false, false),
        ("catalogo:0308", "Aperturas con mancuernas", "pecho", 2, nil, false, false),
        ("catalogo:0251", "Fondos en paralelas", "pecho", 3, "gym", false, false),
        ("catalogo:0652", "Dominadas", "vertical", 3, "gym", false, false),
        ("catalogo:1326", "Dominadas supinas", "vertical", 3, "gym", false, false),
        ("catalogo:0198", "Jalón al pecho en polea", "vertical", 1, nil, false, false),
        ("catalogo:0861", "Remo sentado en polea", "horizontal", 1, nil, false, false),
        ("catalogo:0293", "Remo inclinado con mancuernas", "horizontal", 2, nil, false, false),
        ("catalogo:0027", "Remo con barra", "horizontal", 2, nil, false, false),
        ("catalogo:0499", "Remo invertido", "horizontal", 2, "gym", false, false),
        ("free_exercise_db:Band_Pull_Apart", "Separación de banda", "horizontal", 1, nil, false, false),
        ("free_exercise_db:Face_Pull", "Face pull en polea", "horizontal", 2, nil, false, false),
        ("catalogo:0405", "Press de hombros con mancuernas", "hombro", 2, nil, false, false),
        ("catalogo:0091", "Press militar con barra", "hombro", 2, nil, false, false),
        ("catalogo:0334", "Elevaciones laterales", "hombro", 1, nil, false, false),
        ("catalogo:0406", "Encogimientos con mancuernas", "hombro", 1, nil, false, false),
        ("catalogo:0294", "Curl de bíceps con mancuernas", "biceps", 1, nil, false, false),
        ("catalogo:0031", "Curl de bíceps con barra", "biceps", 1, nil, false, false),
        ("catalogo:0313", "Curl martillo", "biceps", 1, nil, false, false),
        ("free_exercise_db:Triceps_Pushdown", "Extensión de tríceps en polea", "triceps", 1, nil, false, false),
        ("catalogo:0129", "Fondos en banco", "triceps", 1, nil, false, false),
        ("catalogo:0351", "Extensión de tríceps tumbado", "triceps", 2, nil, false, false),
        ("free_exercise_db:Plank", "Plancha", "core", 1, nil, false, true),
        ("wger:580", "Plancha lateral", "core", 1, nil, false, true),
        ("catalogo:0276", "Dead bug", "core", 1, nil, false, false),
        ("wger:1572", "Bird dog (cuadrupedia)", "core", 1, nil, false, false),
        ("wger:1412", "Abdominal bicicleta", "core", 1, nil, false, false),
        ("catalogo:0472", "Elevación de piernas colgado", "core", 3, "gym", false, false),
        ("catalogo:0687", "Giro ruso", "core", 2, nil, false, false),
        ("wger:145", "Leñador en polea", "core", 2, nil, false, false),
        ("free_exercise_db:Pallof_Press", "Press Pallof", "core", 2, nil, false, false),
        ("catalogo:0630", "Escalador", "core", 2, nil, true, false),
        ("catalogo:1373", "Elevación de talones de pie", "pantorrilla", 1, nil, false, false),
        ("catalogo:0605", "Elevación de talones en máquina", "pantorrilla", 1, nil, false, false),
        ("catalogo:0594", "Elevación de talones sentado", "pantorrilla", 1, nil, false, false),
        ("catalogo:0514", "Sentadilla con salto", "potencia", 2, nil, true, false),
        ("catalogo:3361", "Saltos de patinador", "potencia", 2, nil, true, false),
        ("free_exercise_db:Box_Jump_Multiple_Response", "Saltos al cajón", "potencia", 2, "gym", true, false),
        ("catalogo:1160", "Burpee", "potencia", 2, nil, true, false),
        ("wger:1314", "Jumping jacks", "potencia", 1, nil, true, false),
    ]

    /// Familia de equipamiento del pipeline → lo que la persona marca que tiene.
    public static let categoria: [String: String] = [
        "peso corporal": "pc", "peso libre": "pesas", "maquina": "gym", "cable": "gym", "asistido": "gym",
        "maquina de cardio": "gym", "banda elastica": "banda", "balon o rodillo": "balon", "otro": "gym",
    ]

    public static let musculosDelGrupo: [String: [String]] = [
        "rodilla": ["quadriceps"], "cadera": ["glutes", "hamstrings"], "pecho": ["pectorals", "chest"],
        "vertical": ["latissimus dorsi"], "horizontal": ["upper back", "rhomboids", "rear deltoids"],
        "hombro": ["deltoids", "shoulders"], "biceps": ["biceps"], "triceps": ["triceps"],
        "core": ["abdominals", "obliques"], "pantorrilla": ["calves"], "potencia": ["quadriceps", "glutes"],
    ]

    public static let nombreGrupo: [String: String] = [
        "rodilla": "Piernas (cuádriceps)", "cadera": "Glúteos e isquiotibiales", "pecho": "Pecho",
        "vertical": "Espalda (dorsal)", "horizontal": "Espalda alta", "hombro": "Hombros", "biceps": "Bíceps",
        "triceps": "Tríceps", "core": "Core / abdomen", "pantorrilla": "Pantorrillas", "potencia": "Potencia y saltos",
    ]

    /// Deporte → nombre y ejercicios complementarios: [(grupo, ejercicios preferidos)].
    public static let deportes: [(clave: String, nombre: String, complementos: [(String, [String])])] = [
        ("ninguno", "Ninguno / solo gimnasio", []),
        ("futbol", "Fútbol", [("cadera", ["free_exercise_db:Lying_Leg_Curls", "catalogo:1459", "free_exercise_db:Single_Leg_Glute_Bridge"]),
                             ("potencia", ["catalogo:3361", "catalogo:0514"]),
                             ("core", ["free_exercise_db:Pallof_Press", "wger:580"])]),
        ("running", "Running / trote", [("pantorrilla", ["catalogo:1373", "catalogo:0594"]),
                                        ("cadera", ["wger:1823", "free_exercise_db:Single_Leg_Glute_Bridge"]),
                                        ("rodilla", ["wger:999", "catalogo:0431"])]),
        ("ciclismo", "Ciclismo", [("cadera", ["wger:265", "catalogo:1459"]),
                                  ("core", ["wger:1572", "free_exercise_db:Plank"]),
                                  ("horizontal", ["free_exercise_db:Band_Pull_Apart", "catalogo:0293"])]),
        ("natacion", "Natación", [("horizontal", ["free_exercise_db:Face_Pull", "free_exercise_db:Band_Pull_Apart"]),
                                  ("core", ["catalogo:0276", "free_exercise_db:Plank"]),
                                  ("vertical", ["catalogo:0198", "catalogo:0652"])]),
        ("basquet", "Básquetbol / vóleibol", [("potencia", ["free_exercise_db:Box_Jump_Multiple_Response", "catalogo:0514"]),
                                              ("pantorrilla", ["catalogo:1373", "catalogo:0605"]),
                                              ("core", ["wger:580", "free_exercise_db:Pallof_Press"])]),
        ("tenis", "Tenis / pádel", [("core", ["wger:145", "free_exercise_db:Pallof_Press"]),
                                    ("horizontal", ["free_exercise_db:Face_Pull", "free_exercise_db:Band_Pull_Apart"]),
                                    ("rodilla", ["wger:999", "catalogo:0336"])]),
        ("combate", "Artes marciales / boxeo", [("core", ["catalogo:0687", "wger:145"]),
                                                ("cadera", ["free_exercise_db:Single_Leg_Glute_Bridge", "catalogo:0549"]),
                                                ("potencia", ["catalogo:1160", "catalogo:3361"])]),
        ("escalada", "Escalada", [("horizontal", ["free_exercise_db:Face_Pull", "free_exercise_db:Band_Pull_Apart"]),
                                  ("vertical", ["catalogo:0652", "catalogo:0198"]),
                                  ("core", ["catalogo:0276", "catalogo:0472"])]),
        ("otro", "Otro deporte", [("core", ["free_exercise_db:Pallof_Press", "wger:580"]),
                                  ("cadera", ["free_exercise_db:Single_Leg_Glute_Bridge", "wger:265"])]),
    ]

    public struct Prescripcion: Hashable, Sendable {
        public var series: Int
        public var reps: String
        public var descanso: Int
    }

    public static let objetivos: [(clave: String, nombre: String, pres: Prescripcion)] = [
        ("salud", "Salud general", Prescripcion(series: 3, reps: "10–12", descanso: 75)),
        ("fuerza", "Ganar fuerza", Prescripcion(series: 4, reps: "5–6", descanso: 150)),
        ("musculo", "Ganar músculo", Prescripcion(series: 3, reps: "8–12", descanso: 90)),
        ("resistencia", "Resistencia", Prescripcion(series: 3, reps: "15–20", descanso: 45)),
    ]

    public static let equipos: [(clave: String, nombre: String)] = [
        ("pc", "Peso corporal"), ("pesas", "Mancuernas / barras / kettlebell"), ("gym", "Máquinas y poleas (gimnasio)"),
        ("banda", "Bandas elásticas"), ("balon", "Balón suizo"),
    ]

    public static let plantillas: [String: [(String, [String])]] = [
        "full": [
            ("Cuerpo completo A", ["rodilla", "pecho", "horizontal", "cadera", "hombro", "core", "triceps", "pantorrilla"]),
            ("Cuerpo completo B", ["cadera", "vertical", "pecho", "rodilla", "core", "biceps", "hombro", "pantorrilla"]),
            ("Cuerpo completo C", ["rodilla", "hombro", "horizontal", "cadera", "core", "pecho", "biceps", "triceps"]),
        ],
        "tp": [
            ("Torso A", ["pecho", "horizontal", "hombro", "vertical", "triceps", "biceps", "core"]),
            ("Piernas A", ["rodilla", "cadera", "rodilla", "cadera", "pantorrilla", "core", "core"]),
            ("Torso B", ["vertical", "pecho", "horizontal", "hombro", "biceps", "triceps", "core"]),
            ("Piernas B", ["cadera", "rodilla", "cadera", "rodilla", "pantorrilla", "core", "core"]),
        ],
        "epp": [
            ("Empuje", ["pecho", "hombro", "pecho", "triceps", "hombro", "core", "triceps"]),
            ("Tracción", ["vertical", "horizontal", "horizontal", "biceps", "cadera", "core", "biceps"]),
            ("Piernas", ["rodilla", "cadera", "rodilla", "cadera", "pantorrilla", "core", "core"]),
        ],
        "arnold": [
            ("Pecho y espalda", ["pecho", "horizontal", "vertical", "pecho", "horizontal", "core", "core"]),
            ("Hombros y brazos", ["hombro", "biceps", "triceps", "hombro", "biceps", "triceps", "core"]),
            ("Piernas", ["rodilla", "cadera", "rodilla", "cadera", "pantorrilla", "core", "core"]),
        ],
        "weider": [
            ("Pecho", ["pecho", "pecho", "pecho", "triceps", "core", "pecho", "core"]),
            ("Espalda", ["vertical", "horizontal", "horizontal", "vertical", "biceps", "core", "horizontal"]),
            ("Piernas", ["rodilla", "cadera", "rodilla", "cadera", "pantorrilla", "core", "pantorrilla"]),
            ("Hombros", ["hombro", "hombro", "horizontal", "hombro", "core", "core", "hombro"]),
            ("Brazos", ["biceps", "triceps", "biceps", "triceps", "biceps", "triceps", "core"]),
        ],
    ]

    public static let tipos: [(clave: String, nombre: String, detalle: String)] = [
        ("auto", "Recomendada para ti", "Elegimos el tipo según tus días, nivel y edad."),
        ("full", "Cuerpo completo", "Todo el cuerpo en cada sesión."),
        ("tp", "Torso / Pierna", "Un día tren superior y otro tren inferior."),
        ("epp", "Push / Pull / Legs", "Empuje (pecho, hombros, tríceps), tracción (espalda, bíceps) y piernas."),
        ("arnold", "Arnold", "Pecho y espalda, hombros y brazos, y piernas."),
        ("weider", "Un grupo por día", "Pecho, espalda, piernas, hombros y brazos, uno por sesión (Weider)."),
    ]

    public static let minDias: [String: Int] = ["full": 2, "tp": 2, "epp": 3, "arnold": 3, "weider": 5]

    public static func nombreTipo(_ clave: String) -> String { tipos.first { $0.clave == clave }?.nombre ?? clave }
    public static func nombreDeporte(_ clave: String) -> String { deportes.first { $0.clave == clave }?.nombre ?? clave }
    public static func nombreObjetivo(_ clave: String) -> String { objetivos.first { $0.clave == clave }?.nombre ?? clave }
}

// MARK: - Generador

public final class GeneradorRutina {
    struct Curado {
        let i: Int
        let es: String
        let grupo: String
        let nivel: Int
        let eq: String
        let imp: Bool
        let iso: Bool
    }

    struct Ajustes {
        var avisos: [Aviso]
        var nivelMax: Int
        var impacto: Bool
        var objetivo: String
        var pres: TablasRutina.Prescripcion
    }

    private let recomendador: Recomendador
    private let curados: [Curado]
    private let curadoPorIndice: [Int: Curado]

    public init(recomendador: Recomendador) {
        self.recomendador = recomendador
        var lista: [Curado] = []
        for c in TablasRutina.curados {
            guard let e = recomendador.ejercicio(id: c.id) else { continue }
            lista.append(Curado(i: e.indice, es: c.es, grupo: c.grupo, nivel: c.nivel,
                                eq: c.eq ?? TablasRutina.categoria[e.equipamiento] ?? "gym", imp: c.imp, iso: c.iso))
        }
        curados = lista
        curadoPorIndice = Dictionary(uniqueKeysWithValues: lista.map { ($0.i, $0) })
    }

    /// Nombre en español de los ejercicios conocidos (para mostrar la rutina).
    public func nombreEspanol(_ id: String) -> String? {
        recomendador.ejercicio(id: id).flatMap { curadoPorIndice[$0.indice]?.es }
    }

    // ── Reglas por perfil ──────────────────────────────────────────────────────────

    func ajustes(_ p: Perfil) -> Ajustes {
        var avisos: [Aviso] = []
        var nivelMax = ["principiante": 1, "intermedio": 2, "avanzado": 3][p.nivel] ?? 1
        var impacto = true
        var objetivo = p.objetivo
        if p.edad < 14 {
            avisos.append(Aviso("amber", "Menores de 14 años:", "entrena con supervisión de un adulto o profesor de educación física. La rutina usa solo ejercicios simples con peso corporal o bandas."))
            nivelMax = 1; objetivo = "salud"
        } else if p.edad < 18 {
            avisos.append(Aviso("amber", "Menores de 18 años:", "prioriza la técnica antes que el peso y entrena con supervisión al usar cargas. Evitamos series de muy pocas repeticiones."))
            nivelMax = min(nivelMax, 2)
            if objetivo == "fuerza" { objetivo = "musculo" }
        }
        if p.edad >= 60 {
            avisos.append(Aviso("teal", "60 años o más:", "evitamos saltos y ejercicios de técnica compleja, y sumamos trabajo de equilibrio. Consulta a tu médico antes de empezar si no entrenabas."))
            impacto = false; nivelMax = min(nivelMax, 2)
        }
        var imc: Double?
        if let peso = p.peso, let altura = p.altura, peso > 0, altura > 0 { imc = peso / pow(altura / 100, 2) }
        if (imc.map { $0 >= 30 } ?? false) || (p.altura == nil && (p.peso ?? 0) >= 110) {
            if impacto {
                avisos.append(Aviso("teal", "Cuidamos tus articulaciones:", "empezamos sin saltos ni ejercicios de impacto. Puedes sumarlos más adelante."))
            }
            impacto = false
        }
        if !p.lesiones.isEmpty {
            let nombres = p.lesiones.map { TablasRutina.nombreLesion($0).lowercased() }.joined(separator: ", ")
            avisos.append(Aviso("red", "Lesiones excluidas:", "ningún ejercicio de esta rutina carga \(nombres). Si sientes dolor, detente."))
        }
        var pres = TablasRutina.objetivos.first { $0.clave == objetivo }?.pres ?? TablasRutina.objetivos[0].pres
        if p.edad < 18 && pres.reps == "5–6" { pres.reps = "8–12" }
        if p.nivel == "principiante" && pres.series > 3 { pres.series = 3 }
        return Ajustes(avisos: avisos, nivelMax: nivelMax, impacto: impacto, objetivo: objetivo, pres: pres)
    }

    func tipoRecomendado(_ p: Perfil) -> (String, String) {
        let cuidado = p.nivel == "principiante" || (p.edad != 0 && (p.edad < 18 || p.edad >= 60))
        if p.dias <= 2 { return ("full", "Con 2 días, cuerpo completo entrena cada músculo dos veces por semana.") }
        if p.dias == 3 {
            return cuidado
                ? ("full", "Con 3 días y " + (p.nivel == "principiante" ? "nivel principiante" : "tu edad")
                   + ", cuerpo completo repite cada movimiento tres veces por semana: se aprende la técnica más rápido.")
                : ("epp", "Con 3 días y experiencia, Push / Pull / Legs da más volumen a cada músculo por sesión.")
        }
        if p.dias == 4 { return ("tp", "Con 4 días, torso y pierna entrena cada músculo dos veces por semana.") }
        return ("epp", "Con \(p.dias) días, Push / Pull / Legs "
                + (p.dias == 6 ? "se repite dos veces: cada músculo dos veces por semana" : "y se repiten empuje y tracción") + ".")
    }

    func elegirTipo(_ p: Perfil) -> (String, String, [Aviso]) {
        guard p.tipo != "auto", let minimo = TablasRutina.minDias[p.tipo] else {
            let (t, porque) = tipoRecomendado(p)
            return (t, porque, [])
        }
        if p.dias < minimo {
            let (t, porque) = tipoRecomendado(p)
            return (t, porque, [Aviso("amber", "\(TablasRutina.nombreTipo(p.tipo)) necesita al menos \(minimo) días:",
                                      "con \(p.dias) usamos \(TablasRutina.nombreTipo(t).lowercased()).")])
        }
        let extra = p.tipo == "epp" && p.dias == 3 && p.nivel == "principiante"
            ? " Siendo principiante, cuerpo completo suele dar mejores resultados al comienzo." : ""
        return (p.tipo, "Elegiste este tipo de rutina." + extra, [])
    }

    func plantilla(_ tipo: String, _ dias: Int) -> [(String, [String])] {
        let todas = TablasRutina.plantillas[tipo] ?? TablasRutina.plantillas["full"]!
        return tipo == "full" ? Array(todas.prefix(max(2, min(3, dias)))) : todas
    }

    static func ejerciciosPorTiempo(_ minutos: Int) -> Int {
        minutos <= 30 ? 4 : minutos <= 45 ? 5 : minutos <= 60 ? 6 : 7
    }

    // ── Elegir ejercicios ──────────────────────────────────────────────────────────

    func esSeguro(_ i: Int, _ vetadas: Set<String>) -> Bool {
        !recomendador.ejercicios[i].articulaciones.contains { vetadas.contains($0) }
    }

    func candidatos(_ grupo: String, _ p: Perfil, _ aj: Ajustes, _ vet: Set<String>) -> [Curado] {
        let eqs = Set(p.equipo + ["pc"])
        return curados.filter {
            $0.grupo == grupo && eqs.contains($0.eq) && $0.nivel <= aj.nivelMax && (aj.impacto || !$0.imp) && esSeguro($0.i, vet)
        }
    }

    private static let conAparato = "suspen|trx|ring|smith|machine|lever|cable|band|ball|bosu|sled|wheel|roller|wall|towel|assisted"
    private static let golpes = "boxing|punch|jab|hook|uppercut|kick|shadow"

    /// Si no queda ningún ejercicio conocido seguro, se busca en el catálogo completo con las mismas reglas.
    func respaldo(_ grupo: String, _ p: Perfil, _ aj: Ajustes, _ vet: Set<String>, _ usados: Set<Int>) -> [Int] {
        let eqs = Set(p.equipo + ["pc"])
        let objetivo = Set(TablasRutina.musculosDelGrupo[grupo] ?? [])
        var salida: [(Double, Int)] = []
        for e in recomendador.ejercicios {
            let i = e.indice
            if usados.contains(i) || e.esEstiramiento || !esSeguro(i, vet) { continue }
            if let c = curadoPorIndice[i], !eqs.contains(c.eq) || c.nivel > aj.nivelMax || (!aj.impacto && c.imp) { continue }
            guard let categoria = TablasRutina.categoria[e.equipamiento], eqs.contains(categoria), e.equipamiento != "otro" else { continue }
            if !e.agonistas.contains(where: { objetivo.contains($0) }) { continue }
            if Self.coincide(e.nombre, Self.conAparato) && !(eqs.contains("gym") || eqs.contains("banda") || eqs.contains("balon")) { continue }
            if Self.coincide(e.nombre, Self.golpes) { continue }
            let complejidad = e.complejidad ?? ""
            if aj.nivelMax == 1 && complejidad != "baja" { continue }
            if aj.nivelMax == 2 && complejidad == "alta" { continue }
            let puntaje = Double(e.fuente == "catalogo" ? 0 : 5) + Double(complejidad == "baja" ? 0 : 3)
                + Double(e.nombre.utf16.count) / 10
            salida.append((puntaje, i))
        }
        // Orden estable, como Array.prototype.sort en JavaScript
        return salida.enumerated().sorted { ($0.element.0, $0.offset) < ($1.element.0, $1.offset) }
            .prefix(8).map { $0.element.1 }
    }

    private static func coincide(_ texto: String, _ patron: String) -> Bool {
        texto.range(of: patron, options: [.regularExpression, .caseInsensitive]) != nil
    }

    func item(_ i: Int, _ grupo: String, _ aj: Ajustes, _ alternativas: [Int], extra: Bool) -> ItemRutina {
        let iso = curadoPorIndice[i]?.iso ?? false
        let liviano = grupo == "core" || grupo == "pantorrilla"
        return ItemRutina(id: recomendador.ejercicios[i].id, grupo: grupo,
                          series: liviano ? min(3, aj.pres.series) : aj.pres.series,
                          reps: iso ? "30–45 s" : grupo == "core" ? "10–15" : grupo == "potencia" ? "6–8" : aj.pres.reps,
                          descanso: liviano ? min(60, aj.pres.descanso) : aj.pres.descanso,
                          iso: iso, extra: extra,
                          alternativas: alternativas.prefix(8).map { recomendador.ejercicios[$0].id })
    }

    // ── La rutina ──────────────────────────────────────────────────────────────────

    public func generar(_ p: Perfil, hoy: String) -> Rutina {
        var aj = ajustes(p)
        let vet = TablasRutina.vetadas(p.lesiones)
        let (tipo, porqueTipo, avisosTipo) = elegirTipo(p)
        aj.avisos += avisosTipo
        let pl = plantilla(tipo, p.dias)
        let n = Self.ejerciciosPorTiempo(p.minutos)
        var usadosSemana: [Int: Int] = [:]
        var huecos: [String] = []
        var dias: [(titulo: String, equilibrio: Bool, items: [(i: Int, item: ItemRutina)])] = []

        for d in 0..<p.dias {
            let (titulo, grupos) = pl[d % pl.count]
            var usadosDia = Set<Int>()
            var items: [(i: Int, item: ItemRutina)] = []
            for grupo in grupos {
                if items.count >= n { break }
                var cands = candidatos(grupo, p, aj, vet)
                if cands.isEmpty && grupo == "vertical" { cands = candidatos("horizontal", p, aj, vet) }
                var elegido: Int?
                var alternativas: [Int] = []
                if !cands.isEmpty {
                    let ordenados = cands.enumerated().sorted { a, b in
                        let ua = usadosSemana[a.element.i] ?? 0, ub = usadosSemana[b.element.i] ?? 0
                        let na = abs(aj.nivelMax - a.element.nivel), nb = abs(aj.nivelMax - b.element.nivel)
                        return (ua, na, a.offset) < (ub, nb, b.offset)
                    }.map(\.element)
                    if let libre = ordenados.first(where: { !usadosDia.contains($0.i) }) {
                        elegido = libre.i
                        alternativas = ordenados.map(\.i).filter { $0 != libre.i }
                    }
                }
                if elegido == nil {
                    let r = respaldo(grupo, p, aj, vet, usadosDia)
                    if let primero = r.first { elegido = primero; alternativas = Array(r.dropFirst()) }
                }
                guard let i = elegido else {
                    if !huecos.contains(grupo) { huecos.append(grupo) }
                    continue
                }
                usadosDia.insert(i)
                usadosSemana[i, default: 0] += 1
                items.append((i: i, item: item(i, grupo, aj, alternativas, extra: false)))
            }
            let repetido = p.dias > pl.count && d >= pl.count
            dias.append((titulo: repetido ? titulo + " (repetición)" : titulo, equilibrio: false, items: items))
        }

        // Complemento del deporte: un ejercicio extra por complemento, repartidos desde el día 1
        let complementos = TablasRutina.deportes.first { $0.clave == p.deporte }?.complementos ?? []
        for (k, (grupo, preferidos)) in complementos.enumerated() {
            let d = k % dias.count
            let cands = candidatos(grupo, p, aj, vet).map(\.i)
            let usados = Set(dias[d].items.map(\.i))
            let preferido = preferidos.compactMap { recomendador.ejercicio(id: $0)?.indice }
                .first { cands.contains($0) && !usados.contains($0) }
            if let i = preferido ?? cands.first(where: { !usados.contains($0) }) {
                dias[d].items.append((i: i, item: item(i, grupo, aj, cands.filter { $0 != i }, extra: true)))
            }
        }
        if !aj.impacto && p.edad >= 60 {
            for d in dias.indices { dias[d].equilibrio = true }
        }
        let llenos = dias.filter { !$0.items.isEmpty }
        if llenos.count < dias.count {
            aj.avisos.append(llenos.isEmpty
                ? Aviso("amber", "No pudimos armar una rutina segura:", "con tus lesiones y equipamiento ningún ejercicio del catálogo es seguro. Consulta a un kinesiólogo para un plan de rehabilitación.")
                : Aviso("amber", "Tu rutina queda en \(llenos.count) día\(llenos.count > 1 ? "s" : ""):", "con tus lesiones y equipamiento no hay ejercicios seguros suficientes para más días. Sumar equipamiento puede abrir más opciones."))
        }
        return Rutina(creada: hoy, perfil: p, objetivo: aj.objetivo, tipo: tipo, porqueTipo: porqueTipo, avisos: aj.avisos,
                      huecos: huecos,
                      dias: llenos.map { DiaRutina(titulo: $0.titulo, equilibrio: $0.equilibrio, ejercicios: $0.items.map(\.item)) })
    }

    /// Cambia un ejercicio por la siguiente alternativa segura (el de antes pasa al final de la lista).
    public static func cambiar(_ rutina: inout Rutina, dia: Int, posicion: Int) {
        guard rutina.dias.indices.contains(dia), rutina.dias[dia].ejercicios.indices.contains(posicion) else { return }
        var e = rutina.dias[dia].ejercicios[posicion]
        guard !e.alternativas.isEmpty else { return }
        let siguiente = e.alternativas.removeFirst()
        e.alternativas.append(e.id)
        e.id = siguiente
        rutina.dias[dia].ejercicios[posicion] = e
    }

    /// Otra variante de la misma rutina: cada ejercicio se cambia por una de sus tres primeras
    /// alternativas, sin repetir un ejercicio en el mismo día (como «Generar otra variante» en la web).
    public static func variante<G: RandomNumberGenerator>(de rutina: Rutina, generador: inout G) -> Rutina {
        var r = rutina
        for d in r.dias.indices {
            for k in r.dias[d].ejercicios.indices {
                var e = r.dias[d].ejercicios[k]
                guard !e.alternativas.isEmpty else { continue }
                let siguiente = e.alternativas[Int.random(in: 0..<min(3, e.alternativas.count), using: &generador)]
                e.alternativas = e.alternativas.filter { $0 != siguiente } + [e.id]
                e.id = siguiente
                r.dias[d].ejercicios[k] = e
            }
            var vistos = Set<String>()
            for k in r.dias[d].ejercicios.indices {
                var e = r.dias[d].ejercicios[k]
                if vistos.contains(e.id), let otra = e.alternativas.first(where: { !vistos.contains($0) }) {
                    e.alternativas = e.alternativas.filter { $0 != otra } + [e.id]
                    e.id = otra
                    r.dias[d].ejercicios[k] = e
                }
                vistos.insert(e.id)
            }
        }
        return r
    }
}
