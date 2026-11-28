import SwiftUI
import SustitutoSeguroCore

/// Mi cuenta: la misma de la página web. Crear cuenta, iniciar sesión, sincronizar, cerrar
/// sesión y borrar la cuenta con todos sus datos.
struct CuentaView: View {
    @Environment(Cuenta.self) private var cuenta
    @Environment(Almacen.self) private var almacen
    @State private var modo = 0            // 0 = iniciar sesión, 1 = crear cuenta
    @State private var nombre = ""
    @State private var email = ""
    @State private var contrasena = ""
    @State private var acepto = false
    @State private var trabajando = false
    @State private var confirmarSalida = false
    @State private var borrar = false
    @State private var claveBorrado = ""
    @State private var estadoCuenta: ClienteCuentas.EstadoCuenta?

    var body: some View {
        @Bindable var cuenta = cuenta
        Form {
            if let aviso = cuenta.aviso {
                Section { Label(aviso, systemImage: "checkmark.circle").foregroundStyle(.green) }
            }
            if let error = cuenta.error {
                Section { Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red) }
            }
            if let u = cuenta.usuario {
                conectada(u)
            } else {
                Section {
                    TextField("https://…sslip.io", text: $cuenta.servidor)
                        .sinMayusculaInicial().autocorrectionDisabled()
                        .tecladoURL()
                } header: {
                    Text("Servidor")
                } footer: {
                    Text("La dirección de la página (en AWS, la salida «DireccionPagina»). La app usa la misma cuenta que la web.")
                }
                Section {
                    Picker("", selection: $modo) {
                        Text("Iniciar sesión").tag(0)
                        Text("Crear cuenta").tag(1)
                    }
                    .pickerStyle(.segmented)
                    if modo == 1 { TextField("Nombre", text: $nombre).textContentType(.name) }
                    TextField("Correo", text: $email).textContentType(.emailAddress).sinMayusculaInicial()
                        .autocorrectionDisabled().tecladoCorreo()
                    SecureField("Contraseña (mínimo 8 caracteres)", text: $contrasena)
                        .textContentType(modo == 1 ? .newPassword : .password)
                    if modo == 1 {
                        Toggle("Acepto que mis datos de entrenamiento (perfil, rutina, sesiones, dolor y peso) se guarden en el servidor del proyecto. Puedo borrar mi cuenta cuando quiera.", isOn: $acepto)
                            .font(.footnote)
                    }
                    Button {
                        Task { await enviar() }
                    } label: {
                        HStack {
                            Text(modo == 1 ? "Crear cuenta" : "Iniciar sesión").bold()
                            if trabajando { Spacer(); ProgressView() }
                        }
                    }
                    .disabled(trabajando || email.isEmpty || contrasena.count < 8 || (modo == 1 && (nombre.isEmpty || !acepto)))
                } footer: {
                    Text("Sin cuenta, todo queda solo en este teléfono. Al entrar, lo que ya tenías aquí se sube a tu cuenta.")
                }
            }
        }
        .navigationTitle("Mi cuenta")
        .onDisappear { cuenta.aviso = nil }
    }

    @ViewBuilder
    private func conectada(_ u: Usuario) -> some View {
        Section {
            LabeledContent("Nombre", value: u.nombre)
            LabeledContent("Correo", value: u.email)
            LabeledContent("Servidor", value: cuenta.servidor)
            LabeledContent("Estado") {
                switch cuenta.estado {
                case .sinConexion: Text("Sin conexión: se sube al volver").foregroundStyle(.orange)
                case .conectando: ProgressView()
                default: Text("Sincronizada").foregroundStyle(.green)
                }
            }
            if let fecha = cuenta.ultimaSincronizacion {
                LabeledContent("Última sincronización", value: fecha.formatted(date: .omitted, time: .shortened))
            }
        } header: {
            Text("Tu cuenta")
        } footer: {
            Text("Cada cambio se guarda en tu cuenta unos segundos después. Lo ves igual en la página web.")
        }
        if let r = estadoCuenta?.resumen {
            Section("Guardado en el servidor") {
                LabeledContent("Sesiones", value: "\(r["sesiones"] ?? 0)")
                LabeledContent("Registros de peso", value: "\(r["pesos"] ?? 0)")
                LabeledContent("Aportes al modelo", value: "\((r["correcciones"] ?? 0) + (r["revisiones"] ?? 0))")
            }
        }
        Section {
            Button("Sincronizar ahora", systemImage: "arrow.triangle.2.circlepath") {
                Task { trabajando = true; await cuenta.actualizar(); await cargarEstado(); trabajando = false }
            }
            .disabled(trabajando)
            Button("Cerrar sesión", systemImage: "rectangle.portrait.and.arrow.right") { confirmarSalida = true }
        }
        .task { await cargarEstado() }
        .confirmationDialog("¿Cerrar sesión?", isPresented: $confirmarSalida, titleVisibility: .visible) {
            Button("Cerrar sesión", role: .destructive) { cuenta.salir() }
        } message: {
            Text("Tus datos siguen en tu cuenta y se borran de este teléfono.")
        }
        Section {
            if borrar {
                SecureField("Tu contraseña", text: $claveBorrado)
                Button("Borrar mi cuenta y todos sus datos", role: .destructive) {
                    Task { trabajando = true; await cuenta.borrarCuenta(contrasena: claveBorrado); claveBorrado = ""; borrar = false; trabajando = false }
                }
                .disabled(claveBorrado.isEmpty || trabajando)
                Button("Cancelar") { borrar = false; claveBorrado = "" }
            } else {
                Button("Borrar mi cuenta…", role: .destructive) { borrar = true }
            }
        } footer: {
            Text("Borra del servidor tu cuenta, perfil, rutina, sesiones, pesos y aportes. No se puede deshacer.")
        }
    }

    private func enviar() async {
        trabajando = true
        if modo == 1 {
            await cuenta.registrar(nombre: nombre, email: email, contrasena: contrasena)
        } else {
            await cuenta.iniciarSesion(email: email, contrasena: contrasena)
        }
        if cuenta.conectada { contrasena = ""; await cargarEstado() }
        trabajando = false
    }

    private func cargarEstado() async {
        estadoCuenta = try? await cuenta.estadoCuenta()
    }
}
