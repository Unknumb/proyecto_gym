import SwiftUI
import SustitutoSeguroCore

/// Inicio: qué hace la app, el resumen de la semana y accesos a cada sección (como la portada
/// de la página).
struct InicioView: View {
    @Environment(Almacen.self) private var almacen
    @Environment(Cuenta.self) private var cuenta
    let ir: (Pestana) -> Void

    var body: some View {
        let c = almacen.contenido
        let r = Progreso.resumen(c.entrenamientos, hoy: Fechas.iso(), objetivo: c.perfil?.dias ?? 3)
        List {
            Section {
                VStack(alignment: .leading, spacing: 10) {
                    Text("ENTRENAMIENTO CON LESIONES").font(.caption.bold()).foregroundStyle(.teal)
                    (Text("Entrena lo que puedes,\n") + Text("cuida lo que duele.").foregroundColor(.teal))
                        .font(.largeTitle.bold())
                    Text("Encuentra reemplazos seguros para ejercicios que te molestan, arma una rutina según tu edad, deporte y lesiones, y registra cómo avanzas sesión a sesión.")
                        .foregroundStyle(.secondary)
                    HStack {
                        Button(c.rutina == nil ? "Crear mi rutina" : (c.enCurso == nil ? "Entrenar hoy" : "Seguir la sesión")) {
                            ir(c.rutina == nil ? .rutina : .entrenar)
                        }
                        .buttonStyle(.borderedProminent)
                        NavigationLink("Cómo funciona") { ComoFuncionaView() }.buttonStyle(.bordered)
                    }
                }
                .padding(.vertical, 6)
            }

            if r.total > 0 {
                Section("Tu semana") {
                    LabeledContent("Sesiones esta semana", value: "\(r.estaSemana)/\(r.objetivo)")
                    LabeledContent("Semanas seguidas entrenando", value: "\(r.racha)")
                    LabeledContent("Sesiones registradas", value: "\(r.total)")
                }
            }

            Section("Qué puedes hacer") {
                Entrada(icono: "arrow.triangle.swap", titulo: "Reemplazar un ejercicio",
                        texto: "Dinos qué ejercicio no puedes hacer y qué te duele. Te damos alternativas que trabajan el mismo músculo sin cargar esa zona.") {
                    SustituirView()
                }
                Button { ir(.rutina) } label: {
                    EntradaEtiqueta(icono: "list.bullet.clipboard", titulo: "Rutina personalizada",
                                    texto: "Según tu edad, peso, deporte, nivel, tiempo y equipamiento. Las lesiones se excluyen siempre.")
                }
                .buttonStyle(.plain)
                Button { ir(.entrenar) } label: {
                    EntradaEtiqueta(icono: "figure.strengthtraining.traditional", titulo: "Modo entrenamiento",
                                    texto: "Marca cada serie, anota el peso y descansa con cronómetro. Al final registras cómo se sintió tu lesión.")
                }
                .buttonStyle(.plain)
                Button { ir(.progreso) } label: {
                    EntradaEtiqueta(icono: "chart.xyaxis.line", titulo: "Tu progreso",
                                    texto: "Constancia, cargas por ejercicio y evolución del dolor. Si el dolor sube, te avisamos.")
                }
                .buttonStyle(.plain)
                Entrada(icono: "person.crop.circle", titulo: cuenta.usuario.map { "Hola, \($0.nombre)" } ?? "Tu cuenta",
                        texto: cuenta.conectada ? "Tus datos se guardan en tu cuenta y los ves igual en la página web."
                            : "Opcional: crea una cuenta para guardar tu rutina y tu progreso, y verlos también en la web.") {
                    CuentaView()
                }
            }

            Section {
                AvisoView(aviso: Aviso("amber", "Herramienta informativa.", "No reemplaza la evaluación de un kinesiólogo o médico. Si un ejercicio te causa dolor agudo, detente y consulta."))
            }
        }
        .navigationTitle("Sustituto Seguro")
        .tituloEnLinea()
    }
}

private struct Entrada<Destino: View>: View {
    let icono: String
    let titulo: String
    let texto: String
    @ViewBuilder let destino: () -> Destino

    var body: some View {
        NavigationLink { destino() } label: { EntradaEtiqueta(icono: icono, titulo: titulo, texto: texto) }
    }
}

private struct EntradaEtiqueta: View {
    let icono: String
    let titulo: String
    let texto: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icono).font(.title2).foregroundStyle(.teal).frame(width: 32)
            VStack(alignment: .leading, spacing: 3) {
                Text(titulo).font(.headline)
                Text(texto).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 3)
        .contentShape(Rectangle())
    }
}

/// Más: el resto de las secciones.
struct MasView: View {
    @Environment(Cuenta.self) private var cuenta

    var body: some View {
        List {
            Section {
                NavigationLink { CuentaView() } label: {
                    Label(cuenta.usuario?.nombre ?? "Mi cuenta", systemImage: cuenta.conectada ? "person.crop.circle.badge.checkmark" : "person.crop.circle")
                }
            }
            Section("Enseña al modelo") {
                NavigationLink { SustituirView() } label: { Label("Sustituir un ejercicio", systemImage: "arrow.triangle.swap") }
                NavigationLink { NuevoEjercicioView() } label: { Label("Nuevo ejercicio", systemImage: "plus.square.on.square") }
                NavigationLink { MisDatosView() } label: { Label("Mis aportes y exportación", systemImage: "tray.full") }
            }
            Section {
                NavigationLink { ComoFuncionaView() } label: { Label("Cómo funciona", systemImage: "questionmark.circle") }
            }
        }
        .navigationTitle("Más")
    }
}

/// Cómo funciona: lo mismo que explica la página.
struct ComoFuncionaView: View {
    private let preguntas: [(String, String)] = [
        ("¿Cómo sabe qué ejercicio es seguro?",
         "Cada ejercicio tiene registrados sus músculos principales y de apoyo, y cada músculo las articulaciones que cruza. Si un ejercicio carga una articulación que marcaste, queda fuera. Es un filtro estricto: no importa qué tan bueno sea el ejercicio.\n\nPara lesiones usamos un criterio conservador. Por ejemplo, si marcas muñeca también evitamos los ejercicios que cargan el antebrazo, y si marcas zona lumbar evitamos los que cargan la espalda media."),
        ("¿Por qué a veces me cambia el músculo?",
         "Algunos músculos cruzan la articulación lesionada. El cuádriceps cruza la rodilla: no existe forma de trabajarlo sin cargarla. En esos casos mantenemos los músculos que sí se pueden trabajar (por ejemplo, glúteos) y te lo explicamos.\n\nSi ningún músculo principal se puede conservar, la aplicación no recomienda nada y te sugiere consultar a un kinesiólogo, en vez de inventar una alternativa."),
        ("¿Cómo se ordenan las alternativas?",
         "Primero las del mismo tipo de movimiento (empuje, tracción, estiramiento). Después por parecido biomecánico, calculado con un grafo que conecta ejercicios, músculos y articulaciones (Node2Vec, agrupado en 12 familias de movimiento)."),
        ("¿Cómo se arma la rutina?",
         "Puedes elegir el tipo: cuerpo completo, torso / pierna, Push / Pull / Legs, Arnold o un grupo por día. Si dejas «Recomendada para ti», lo elegimos según tus días y tu nivel: con 2 días, cuerpo completo; con 3, cuerpo completo para principiantes y Push / Pull / Legs con experiencia; con 4, torso y pierna; con 5 o 6, Push / Pull / Legs. Los minutos definen cuántos ejercicios entran. El objetivo define series, repeticiones y descanso.\n\nLa edad ajusta la dificultad: con menos de 18 años se evitan series de muy pocas repeticiones; con 60 o más se evitan saltos y se suma equilibrio. El peso y la altura sirven para decidir si conviene empezar sin ejercicios de impacto. El deporte agrega ejercicios complementarios. Es el mismo generador de la página web: con los mismos datos arma la misma rutina."),
        ("¿El modelo aprende de mí?",
         "Sí, si quieres. Cuando corriges cómo describimos un ejercicio o calificas una alternativa, se guarda como aporte. Con cuenta, el aporte llega al servidor; sin cuenta, lo exportas en «Mis aportes y exportación». El equipo reentrena con los aportes de todos, y solo los adopta si el modelo no empeora."),
        ("¿Dónde quedan mis datos?",
         "Sin cuenta, solo en este teléfono. Con cuenta, también en el servidor del proyecto (una base de datos PostgreSQL en AWS), y los ves igual en la página web. La contraseña se guarda cifrada (hash), y puedes borrar tu cuenta con todos sus datos desde «Mi cuenta». Al cerrar sesión, los datos se borran del teléfono y siguen en tu cuenta."),
        ("¿Qué tan confiable es?",
         "Con 2.461 ejercicios de tres fuentes abiertas, el sistema encuentra al menos 3 alternativas para el 89 % de los ejercicios que se pueden sustituir. En rodilla y hombro la cobertura es menor.\n\nEs una herramienta informativa basada en datos: no conoce tu lesión en detalle. Un kinesiólogo siempre tiene la última palabra."),
    ]

    var body: some View {
        List {
            Section {
                Text("Esta aplicación nace de un proyecto de Machine Learning que modela cada ejercicio por su mecánica: qué músculos trabaja y qué articulaciones carga. Así puede evitar la zona lesionada sin dejar de entrenar el músculo.")
            }
            Section("Cómo usarla") {
                Paso(n: 1, titulo: "Arma tu rutina", texto: "Ingresa edad, peso, deporte, nivel, días, minutos, equipamiento y lesiones. La rutina se crea al instante.")
                Paso(n: 2, titulo: "Ajústala", texto: "Cambia un ejercicio por otra opción segura deslizándolo, o genera otra variante.")
                Paso(n: 3, titulo: "Entrena con la app", texto: "Elige el día, anota peso y repeticiones, marca cada serie y descansa con el cronómetro.")
                Paso(n: 4, titulo: "Registra cómo te sentiste", texto: "Al terminar anotas el esfuerzo y el dolor de cada zona lesionada. Si sube, la app te avisa.")
                Paso(n: 5, titulo: "Mira tu progreso", texto: "Constancia semanal, cargas por ejercicio, dolor y peso corporal en gráficos.")
            }
            Section("Preguntas frecuentes") {
                ForEach(preguntas.indices, id: \.self) { k in
                    DisclosureGroup(preguntas[k].0) { Text(preguntas[k].1).font(.subheadline).foregroundStyle(.secondary) }
                }
            }
        }
        .navigationTitle("Cómo funciona")
    }
}

private struct Paso: View {
    let n: Int
    let titulo: String
    let texto: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(n)").font(.headline).frame(width: 28, height: 28).background(.teal.opacity(0.2), in: Circle())
            VStack(alignment: .leading, spacing: 2) {
                Text(titulo).font(.headline)
                Text(texto).font(.subheadline).foregroundStyle(.secondary)
            }
        }
    }
}
