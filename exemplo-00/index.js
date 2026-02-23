import tf from '@tensorflow/tfjs-node';


async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    // Primeira camada da rede:
    // Entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)

    // 80 neurônios = aqui coloquei bastante porque são poucos dados para treinamento
    // Quanto mais neurônios, mais complexidade a rede pode aprender e consequentemente
    // Mais processamento vai usar

    // A ReLU age como um filtro é como se ela deixasse apenas os dados interessantes seguirem viagem na rede
    // Se a informação chegou nesse neurônio é positiva passa pra frente!
    // Se for 0 ou negativa, pode jogar fora, não serve para nada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }))

    // Saída 3 neurônios
    // Um para cada categoria [premium, medium, basic]
    // activation: softmax normaliza a saida em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: "softmax" }))

    // Compilando o modelo
    // Optimazer Adam (Adaptive Moment Estimation)
    // É um treinador pessoal moderno para redes neurais
    // Ajusta os pesos de forma eficiente e inteligente 
    // Apender com histórico de erros e acertos

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria) com a resposta certa
    // A categoria "premium" será sempre [1, 0, 0]

    // Quanto mais distante da previsão do modelo da resposta correta, maior o erro (loss)
    // Exemplo classico: classificação de imagens, recomendações e categorização de usuários
    // Qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"
    model.compile({ 
        optimizer: "adam", 
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    })

    // Treinamento do modelo passando entrada e saida
    // verbose: desabilita logs internos (usa só o callback)
    // epochs: quantidade de vezes que vai rodar no dataset
    // shuffle: embaralha os dados para evitar viés (bias)
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                // onEpochEnd: (epochs, log) => console.log(
                //     `Epoch: ${epochs}: loss = ${log.loss}`
                // )
            }
        }
    )

    return model
}

async function predict(model, pessoa) {
    // Transformar o array js para o tensor (tfjs)
    const tfInput = tf.tensor2d(pessoa)

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))

}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dado melhor
// assim o algoritmo consegue entender melhor os padrões complexos dos dados
const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: "Zé", idade: 28, cor: "verde", localizacao: "Curitiba" }
// Normalizando a idade da nova pessoa passando o mesmo padrão do treino
// Exemplo: idade_min = 25, idade_max = 40
// (28 - idade_min) / (idade_max - idade_min) = 0.2

const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        0,    // cor azul
        0,    // cor vermelho
        1,    // cor verde
        0,    // localização São Paulo
        0,    // localização Rio
        1     // localização Curitiba
    ]
]


const predictions = await predict(model, pessoaTensorNormalizado) 
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join("\n")

console.log(results)