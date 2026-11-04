import SwiftUI
import SustitutoSeguroCore

@main
struct SustitutoSeguroApp: App {
    @State private var catalogo = Catalogo()
    @State private var almacen = Almacen()

    var body: some Scene {
        WindowGroup {
            Raiz()
                .environment(catalogo)
                .environment(almacen)
                .task { await catalogo.cargar() }
        }
    }
}

/// Las cuatro secciones de la app. El modelo y las recomendaciones se cargan una vez
/// al abrir; mientras tanto se muestra un indicador.
struct Raiz: View {
    @Environment(Catalogo.self) private var catalogo

    var body: some View {
        if let error = catalogo.error {
            ContentUnavailableView("No se pudieron cargar los datos", systemImage: "exclamationmark.triangle",
                                   description: Text(error))
        } else if catalogo.recomendador == nil || catalogo.perfilador == nil {
            ProgressView("Cargando el modelo…")
        } else {
            TabView {
                NavigationStack { SustituirView() }
                    .tabItem { Label("Sustituir", systemImage: "arrow.triangle.swap") }
                NavigationStack { NuevoEjercicioView() }
                    .tabItem { Label("Nuevo ejercicio", systemImage: "plus.square.on.square") }
                NavigationStack { EntrenarView() }
                    .tabItem { Label("Entrenar", systemImage: "figure.strengthtraining.traditional") }
                NavigationStack { MisDatosView() }
                    .tabItem { Label("Mis datos", systemImage: "tray.full") }
            }
        }
    }
}
