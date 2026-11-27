import SwiftUI
import SustitutoSeguroCore

@main
struct SustitutoSeguroApp: App {
    @State private var catalogo = Catalogo()
    @State private var almacen = Almacen()
    @State private var cuenta = Cuenta()

    var body: some Scene {
        WindowGroup {
            Raiz()
                .environment(catalogo)
                .environment(almacen)
                .environment(cuenta)
                .task {
                    cuenta.vincular(almacen)
                    await catalogo.cargar()
                }
        }
    }
}

enum Pestana: Hashable { case inicio, rutina, entrenar, progreso, mas }

/// Las cinco secciones de la app, las mismas de la página: Inicio, Mi rutina, Entrenar,
/// Progreso y Más (cuenta, sustituir, nuevo ejercicio, aportes y cómo funciona). El modelo y
/// las recomendaciones se cargan una vez al abrir; mientras tanto se muestra un indicador.
struct Raiz: View {
    @Environment(Catalogo.self) private var catalogo
    @Environment(Almacen.self) private var almacen
    @Environment(Cuenta.self) private var cuenta
    @Environment(\.scenePhase) private var fase
    @State private var pestana: Pestana = .inicio

    var body: some View {
        if let error = catalogo.error {
            ContentUnavailableView("No se pudieron cargar los datos", systemImage: "exclamationmark.triangle",
                                   description: Text(error))
        } else if catalogo.recomendador == nil || catalogo.perfilador == nil || catalogo.generador == nil {
            ProgressView("Cargando el modelo…")
        } else {
            TabView(selection: $pestana) {
                NavigationStack { InicioView { pestana = $0 } }
                    .tabItem { Label("Inicio", systemImage: "house") }.tag(Pestana.inicio)
                NavigationStack { RutinaView() }
                    .tabItem { Label("Mi rutina", systemImage: "list.bullet.clipboard") }.tag(Pestana.rutina)
                NavigationStack { EntrenarView() }
                    .tabItem { Label("Entrenar", systemImage: "figure.strengthtraining.traditional") }.tag(Pestana.entrenar)
                NavigationStack { ProgresoView() }
                    .tabItem { Label("Progreso", systemImage: "chart.xyaxis.line") }.tag(Pestana.progreso)
                NavigationStack { MasView() }
                    .tabItem { Label("Más", systemImage: "ellipsis.circle") }.tag(Pestana.mas)
            }
            .tint(.teal)
            .onChange(of: fase) { _, nueva in
                // Al volver a la app se baja lo nuevo de la cuenta (por ejemplo, una sesión de la web)
                if nueva == .active && cuenta.conectada { Task { await cuenta.actualizar() } }
            }
            .onAppear { almacen.nombres = Dictionary(catalogo.recomendador!.ejercicios.map { ($0.id, catalogo.nombre($0.id)) }, uniquingKeysWith: { a, _ in a }) }
        }
    }
}
