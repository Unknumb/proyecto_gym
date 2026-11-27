import Foundation
import Observation
import Security
import SustitutoSeguroCore

/// La cuenta en el servidor (la misma de la página web). Es opcional: sin cuenta todo funciona
/// en el teléfono. Con cuenta:
/// - al entrar se suben los datos guardados sin cuenta (lo que ya está en la cuenta gana) y se
///   baja todo lo de la cuenta;
/// - cada cambio se sube unos segundos después, solo lo que cambió;
/// - al abrir la app se suben los cambios pendientes y se baja lo nuevo (por ejemplo, una
///   sesión registrada en la web);
/// - al cerrar sesión se borran los datos del teléfono (siguen en la cuenta).
///
/// El token va en el llavero (Keychain); la dirección del servidor, en las preferencias.
@MainActor
@Observable
final class Cuenta {
    enum Estado: Equatable { case sinCuenta, conectando, conectada, sinConexion }

    private(set) var usuario: Usuario?
    private(set) var estado: Estado = .sinCuenta
    private(set) var ultimaSincronizacion: Date?
    var error: String?
    var aviso: String?
    var servidor: String {
        didSet { UserDefaults.standard.set(servidor, forKey: Self.claveServidor) }
    }

    @ObservationIgnored private var cliente: ClienteCuentas?
    @ObservationIgnored private var foto: FotoCuenta
    @ObservationIgnored private var subida: Task<Void, Never>?
    @ObservationIgnored private weak var almacen: Almacen?
    @ObservationIgnored private var aplicando = false

    private static let claveServidor = "servidor"
    private static let claveUsuario = "usuario"
    private static let claveFoto = "foto_cuenta"

    init() {
        servidor = UserDefaults.standard.string(forKey: Self.claveServidor) ?? ""
        foto = UserDefaults.standard.data(forKey: Self.claveFoto)
            .flatMap { try? JSONDecoder().decode(FotoCuenta.self, from: $0) } ?? FotoCuenta()
        if let token = Llavero.leer(), let url = ClienteCuentas.direccion(servidor),
           let datos = UserDefaults.standard.data(forKey: Self.claveUsuario),
           let u = try? JSONDecoder().decode(Usuario.self, from: datos) {
            cliente = ClienteCuentas(servidor: url, token: token)
            usuario = u
            estado = .conectada
        }
    }

    var conectada: Bool { usuario != nil }

    /// Se llama una vez al abrir la app.
    func vincular(_ almacen: Almacen) {
        self.almacen = almacen
        almacen.alCambiar = { [weak self] in MainActor.assumeIsolated { self?.programarSubida() } }
        if conectada { Task { await actualizar() } }
    }

    // MARK: Entrar y salir

    func registrar(nombre: String, email: String, contrasena: String) async {
        await entrar { try await $0.registrar(nombre: nombre, email: email, contrasena: contrasena) }
    }

    func iniciarSesion(email: String, contrasena: String) async {
        await entrar { try await $0.iniciarSesion(email: email, contrasena: contrasena) }
    }

    private func entrar(_ accion: (ClienteCuentas) async throws -> ClienteCuentas.Entrada) async {
        guard let url = ClienteCuentas.direccion(servidor) else {
            error = "Escribe la dirección del servidor, por ejemplo https://3.85.12.40.sslip.io"
            return
        }
        guard let almacen else { return }
        error = nil; aviso = nil
        estado = .conectando
        let c = ClienteCuentas(servidor: url)
        do {
            let e = try await accion(c)
            let datos = try await c.sincronizar(almacen.datosCuenta)
            cliente = c
            usuario = e.usuario
            Llavero.guardar(e.token)
            UserDefaults.standard.set(try? JSONEncoder().encode(e.usuario), forKey: Self.claveUsuario)
            servidor = url.absoluteString
            aplicarDelServidor(datos)
            estado = .conectada
        } catch {
            self.error = error.localizedDescription
            estado = .sinCuenta
        }
    }

    /// Cierra la sesión y borra los datos del teléfono (siguen guardados en la cuenta).
    func salir() {
        olvidarSesion()
        almacen?.borrarTodo()
        aviso = "Cerraste sesión. Tus datos siguen en tu cuenta; se borraron de este teléfono."
    }

    func borrarCuenta(contrasena: String) async {
        guard let cliente else { return }
        error = nil
        do {
            try await cliente.borrarCuenta(contrasena: contrasena)
            olvidarSesion()
            almacen?.borrarTodo()
            aviso = "Tu cuenta y todos sus datos se borraron del servidor y de este teléfono."
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func olvidarSesion() {
        subida?.cancel()
        cliente = nil
        usuario = nil
        estado = .sinCuenta
        foto = FotoCuenta()
        ultimaSincronizacion = nil
        Llavero.borrar()
        UserDefaults.standard.removeObject(forKey: Self.claveUsuario)
        UserDefaults.standard.removeObject(forKey: Self.claveFoto)
    }

    func estadoCuenta() async throws -> ClienteCuentas.EstadoCuenta? {
        try await cliente?.cuenta()
    }

    // MARK: Sincronizar

    /// Sube lo pendiente y baja lo de la cuenta.
    func actualizar() async {
        guard let cliente, let almacen else { return }
        error = nil
        do {
            try await subirPendiente(cliente, almacen)
            aplicarDelServidor(try await cliente.datos())
            estado = .conectada
        } catch {
            manejar(error)
        }
    }

    private func programarSubida() {
        guard conectada, !aplicando else { return }
        subida?.cancel()
        subida = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled, let self, let cliente = self.cliente, let almacen = self.almacen else { return }
            do {
                try await self.subirPendiente(cliente, almacen)
                self.estado = .conectada
            } catch {
                self.manejar(error)
            }
        }
    }

    private func subirPendiente(_ cliente: ClienteCuentas, _ almacen: Almacen) async throws {
        let local = almacen.datosCuenta
        let cambios = CambiosCuenta.entre(foto, local)
        guard !cambios.vacio else { return }
        try await cliente.subir(cambios)
        // Lo subido queda como la nueva foto, tal como está en el teléfono
        var f = foto
        if let p = cambios.perfil { f.perfil = FotoCuenta.json(p) }
        if let r = cambios.rutina { f.rutina = FotoCuenta.json(r) }
        if cambios.quitarRutina { f.rutina = nil }
        for s in cambios.sesiones { f.sesiones[s.uid] = FotoCuenta.json(s) }
        for uid in cambios.quitarSesiones { f.sesiones[uid] = nil }
        for p in cambios.pesos { f.pesos[p.fecha] = p.kg }
        for fecha in cambios.quitarPesos { f.pesos[fecha] = nil }
        f.aportes.formUnion(cambios.correcciones.map(FotoCuenta.json) + cambios.revisiones.map(FotoCuenta.json))
        guardarFoto(f)
        ultimaSincronizacion = Date()
    }

    private func aplicarDelServidor(_ datos: DatosCuenta) {
        aplicando = true
        almacen?.aplicar(datos)
        aplicando = false
        var f = FotoCuenta(datos)
        // Los aportes locales ya están en la cuenta después de sincronizar
        f.aportes.formUnion(foto.aportes)
        if let almacen {
            let local = almacen.datosCuenta
            f.aportes.formUnion(local.correcciones.map(FotoCuenta.json) + local.revisiones.map(FotoCuenta.json))
        }
        guardarFoto(f)
        ultimaSincronizacion = Date()
    }

    private func guardarFoto(_ f: FotoCuenta) {
        foto = f
        UserDefaults.standard.set(try? JSONEncoder().encode(f), forKey: Self.claveFoto)
    }

    private func manejar(_ error: Error) {
        if let e = error as? ErrorCuenta, e.sesionVencida {
            // Sesión vencida: los datos quedan en el teléfono y se suben al volver a entrar
            olvidarSesion()
            self.error = "Tu sesión venció. Vuelve a iniciar sesión: lo que guardaste en el teléfono se mantiene."
        } else if let e = error as? ErrorCuenta, e.estado == 0 {
            estado = .sinConexion
        } else {
            self.error = error.localizedDescription
        }
    }
}

/// El token de la sesión en el llavero de iOS: cifrado y solo para esta app.
enum Llavero {
    private static let servicio = "cl.unknumb.sustitutoseguro.cuenta"

    private static var consulta: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: servicio,
         kSecAttrAccount as String: "token"]
    }

    static func guardar(_ token: String) {
        borrar()
        var q = consulta
        q[kSecValueData as String] = Data(token.utf8)
        q[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        SecItemAdd(q as CFDictionary, nil)
    }

    static func leer() -> String? {
        var q = consulta
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var resultado: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &resultado) == errSecSuccess, let datos = resultado as? Data else { return nil }
        return String(data: datos, encoding: .utf8)
    }

    static func borrar() { SecItemDelete(consulta as CFDictionary) }
}
