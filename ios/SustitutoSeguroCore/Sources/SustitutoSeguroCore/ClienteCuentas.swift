import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// Cliente de la API de cuentas (api/servidor.py, README §20): la misma que usa la página.

public struct Usuario: Codable, Hashable, Sendable {
    public var id: Int
    public var email: String
    public var nombre: String
    public var creado: String
}

public struct CorreccionAporte: Codable, Hashable, Sendable {
    public var ejercicio_id: String
    public var objetivo: String
    public var etiqueta: String
    public var etiqueta_previa: String
    public var origen_previo: String
    public var confianza_previa: Double?
    public var fecha: String

    public init(ejercicio_id: String, objetivo: String, etiqueta: String, etiqueta_previa: String,
                origen_previo: String, confianza_previa: Double?, fecha: String) {
        self.ejercicio_id = ejercicio_id; self.objetivo = objetivo; self.etiqueta = etiqueta
        self.etiqueta_previa = etiqueta_previa; self.origen_previo = origen_previo
        self.confianza_previa = confianza_previa; self.fecha = fecha
    }

    /// Una corrección vale una vez por ejercicio y objetivo: la última reemplaza a las anteriores.
    public var clave: String { "\(ejercicio_id)|\(objetivo)" }
}

public struct RevisionAporte: Codable, Hashable, Sendable {
    public var original_id: String
    public var alternativa_id: String
    public var articulacion: String
    public var perfil: String
    public var veredicto: String
    public var similitud: Double?
    public var misma_familia: Bool
    public var fecha: String

    public init(original_id: String, alternativa_id: String, articulacion: String, perfil: String,
                veredicto: String, similitud: Double?, misma_familia: Bool, fecha: String) {
        self.original_id = original_id; self.alternativa_id = alternativa_id; self.articulacion = articulacion
        self.perfil = perfil; self.veredicto = veredicto; self.similitud = similitud
        self.misma_familia = misma_familia; self.fecha = fecha
    }

    public var clave: String { "\(original_id)|\(alternativa_id)|\(articulacion)|\(perfil)" }
}

/// Todo lo que la cuenta guarda de una persona (`GET /api/datos`).
public struct DatosCuenta: Codable, Hashable, Sendable {
    public var perfil: Perfil?
    public var rutina: Rutina?
    public var sesiones: [Sesion]
    public var pesos: [Pesaje]
    public var correcciones: [CorreccionAporte]
    public var revisiones: [RevisionAporte]

    public init(perfil: Perfil? = nil, rutina: Rutina? = nil, sesiones: [Sesion] = [], pesos: [Pesaje] = [],
                correcciones: [CorreccionAporte] = [], revisiones: [RevisionAporte] = []) {
        self.perfil = perfil; self.rutina = rutina; self.sesiones = sesiones; self.pesos = pesos
        self.correcciones = correcciones; self.revisiones = revisiones
    }

    private enum Claves: String, CodingKey { case perfil, rutina, sesiones, pesos, correcciones, revisiones }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Claves.self)
        perfil = try c.decodeIfPresent(Perfil.self, forKey: .perfil)
        rutina = try c.decodeIfPresent(Rutina.self, forKey: .rutina)
        sesiones = try c.decodeIfPresent([Sesion].self, forKey: .sesiones) ?? []
        pesos = try c.decodeIfPresent([Pesaje].self, forKey: .pesos) ?? []
        correcciones = try c.decodeIfPresent([CorreccionAporte].self, forKey: .correcciones) ?? []
        revisiones = try c.decodeIfPresent([RevisionAporte].self, forKey: .revisiones) ?? []
    }
}

/// Lo que ya está en el servidor, para subir solo lo que cambió (como `subirCambios` en la web).
public struct FotoCuenta: Codable, Hashable, Sendable {
    public var perfil: String?
    public var rutina: String?
    public var sesiones: [String: String] = [:]     // uid → JSON
    public var pesos: [String: Double] = [:]        // fecha → kg
    public var aportes: Set<String> = []            // JSON de cada aporte ya subido

    public init() {}

    public init(_ d: DatosCuenta) {
        perfil = d.perfil.map(Self.json)
        rutina = d.rutina.map(Self.json)
        sesiones = Dictionary(d.sesiones.map { ($0.uid, Self.json($0)) }, uniquingKeysWith: { _, b in b })
        pesos = Dictionary(d.pesos.map { ($0.fecha, $0.kg) }, uniquingKeysWith: { _, b in b })
        aportes = Set(d.correcciones.map(Self.json) + d.revisiones.map(Self.json))
    }

    public static func json<T: Encodable>(_ valor: T) -> String {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]
        return (try? e.encode(valor)).flatMap { String(data: $0, encoding: .utf8) } ?? ""
    }
}

/// Cambios entre la foto del servidor y lo que hay en el teléfono.
public struct CambiosCuenta: Hashable, Sendable {
    public var perfil: Perfil?
    public var rutina: Rutina?
    public var quitarRutina = false
    public var sesiones: [Sesion] = []
    public var quitarSesiones: [String] = []
    public var pesos: [Pesaje] = []
    public var quitarPesos: [String] = []
    public var correcciones: [CorreccionAporte] = []
    public var revisiones: [RevisionAporte] = []

    public var vacio: Bool {
        perfil == nil && rutina == nil && !quitarRutina && sesiones.isEmpty && quitarSesiones.isEmpty
            && pesos.isEmpty && quitarPesos.isEmpty && correcciones.isEmpty && revisiones.isEmpty
    }

    public static func entre(_ foto: FotoCuenta, _ local: DatosCuenta) -> CambiosCuenta {
        var c = CambiosCuenta()
        if let p = local.perfil, FotoCuenta.json(p) != foto.perfil { c.perfil = p }
        if let r = local.rutina {
            if FotoCuenta.json(r) != foto.rutina { c.rutina = r }
        } else if foto.rutina != nil {
            c.quitarRutina = true
        }
        let uids = Set(local.sesiones.map(\.uid))
        c.sesiones = local.sesiones.filter { foto.sesiones[$0.uid] != FotoCuenta.json($0) }
        c.quitarSesiones = foto.sesiones.keys.filter { !uids.contains($0) }.sorted()
        let fechas = Set(local.pesos.map(\.fecha))
        c.pesos = local.pesos.filter { foto.pesos[$0.fecha] != $0.kg }
        c.quitarPesos = foto.pesos.keys.filter { !fechas.contains($0) }.sorted()
        c.correcciones = local.correcciones.filter { !foto.aportes.contains(FotoCuenta.json($0)) }
        c.revisiones = local.revisiones.filter { !foto.aportes.contains(FotoCuenta.json($0)) }
        return c
    }
}

public struct ErrorCuenta: Error, LocalizedError, Hashable, Sendable {
    public var mensaje: String
    public var estado: Int          // 0 = sin conexión
    public var errorDescription: String? { mensaje }
    /// La sesión venció o la cuenta ya no existe: hay que volver a iniciar sesión.
    public var sesionVencida: Bool { estado == 401 }
}

public final class ClienteCuentas: @unchecked Sendable {
    public let servidor: URL
    public var token: String?
    private let sesion: URLSession

    public init(servidor: URL, token: String? = nil, sesion: URLSession = .shared) {
        self.servidor = servidor; self.token = token; self.sesion = sesion
    }

    /// Normaliza lo que la persona escribe: agrega https:// y quita la barra final.
    public static func direccion(_ texto: String) -> URL? {
        var t = texto.trimmingCharacters(in: .whitespacesAndNewlines)
        while t.hasSuffix("/") { t.removeLast() }
        if t.isEmpty { return nil }
        if !t.lowercased().hasPrefix("http://") && !t.lowercased().hasPrefix("https://") { t = "https://" + t }
        guard let url = URL(string: t), url.host != nil else { return nil }
        return url
    }

    public struct Entrada: Decodable, Sendable { public let token: String; public let usuario: Usuario }
    public struct EstadoCuenta: Decodable, Sendable { public let usuario: Usuario; public let resumen: [String: Int] }
    private struct Vacio: Decodable {}
    private struct Salud: Decodable { let ok: Bool }

    // MARK: Cuenta

    public func salud() async throws -> Bool { try await pedir("GET", "/api/salud", Salud.self).ok }

    public func registrar(nombre: String, email: String, contrasena: String) async throws -> Entrada {
        let e = try await pedir("POST", "/api/registro", Entrada.self, ["nombre": nombre, "email": email, "contrasena": contrasena])
        token = e.token
        return e
    }

    public func iniciarSesion(email: String, contrasena: String) async throws -> Entrada {
        let e = try await pedir("POST", "/api/sesion", Entrada.self, ["email": email, "contrasena": contrasena])
        token = e.token
        return e
    }

    public func cuenta() async throws -> EstadoCuenta { try await pedir("GET", "/api/cuenta", EstadoCuenta.self) }

    public func borrarCuenta(contrasena: String) async throws {
        _ = try await pedir("DELETE", "/api/cuenta", Vacio.self, ["contrasena": contrasena])
        token = nil
    }

    // MARK: Datos

    public func datos() async throws -> DatosCuenta { try await pedir("GET", "/api/datos", DatosCuenta.self) }

    /// Sube lo guardado sin cuenta; lo que ya está en la cuenta gana. Devuelve todo lo de la cuenta.
    public func sincronizar(_ local: DatosCuenta) async throws -> DatosCuenta {
        try await pedir("POST", "/api/sincronizar", DatosCuenta.self, local)
    }

    public func subir(_ c: CambiosCuenta) async throws {
        if let p = c.perfil { _ = try await pedir("PUT", "/api/datos/perfil", Vacio.self, p) }
        if let r = c.rutina { _ = try await pedir("PUT", "/api/datos/rutina", Vacio.self, r) }
        if c.quitarRutina { _ = try await pedir("DELETE", "/api/datos/rutina", Vacio.self) }
        for s in c.sesiones { _ = try await pedir("POST", "/api/datos/sesiones", Vacio.self, s) }
        for uid in c.quitarSesiones { _ = try await pedir("DELETE", "/api/datos/sesiones/\(Self.ruta(uid))", Vacio.self) }
        for p in c.pesos { _ = try await pedir("PUT", "/api/datos/pesos", Vacio.self, p) }
        for f in c.quitarPesos { _ = try await pedir("DELETE", "/api/datos/pesos/\(Self.ruta(f))", Vacio.self) }
        if !c.correcciones.isEmpty || !c.revisiones.isEmpty {
            struct Aportes: Encodable { let correcciones: [CorreccionAporte]; let revisiones: [RevisionAporte] }
            _ = try await pedir("POST", "/api/datos/aportes", Vacio.self, Aportes(correcciones: c.correcciones, revisiones: c.revisiones))
        }
    }

    private static func ruta(_ s: String) -> String {
        s.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed.subtracting(CharacterSet(charactersIn: "/"))) ?? s
    }

    // MARK: HTTP

    private struct SinCuerpo: Encodable {}
    private struct RespuestaError: Decodable { let error: String }

    private func pedir<R: Decodable>(_ metodo: String, _ ruta: String, _ tipo: R.Type) async throws -> R {
        try await pedir(metodo, ruta, tipo, Optional<SinCuerpo>.none)
    }

    private func pedir<R: Decodable, C: Encodable>(_ metodo: String, _ ruta: String, _ tipo: R.Type, _ cuerpo: C?) async throws -> R {
        var req = URLRequest(url: servidor.appendingPathComponent(String(ruta.dropFirst())), timeoutInterval: 20)
        req.httpMethod = metodo
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let cuerpo {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONEncoder().encode(cuerpo)
        }
        let datos: Data, respuesta: URLResponse
        do {
            (datos, respuesta) = try await sesion.data(for: req)
        } catch {
            throw ErrorCuenta(mensaje: "No se pudo conectar con el servidor. Revisa la dirección y tu conexión.", estado: 0)
        }
        let estado = (respuesta as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(estado) else {
            let mensaje = (try? JSONDecoder().decode(RespuestaError.self, from: datos))?.error
            throw ErrorCuenta(mensaje: mensaje ?? "El servidor respondió con un error (\(estado)).", estado: estado)
        }
        do {
            return try JSONDecoder().decode(R.self, from: datos)
        } catch {
            throw ErrorCuenta(mensaje: "La respuesta del servidor no tiene el formato esperado.", estado: estado)
        }
    }
}
