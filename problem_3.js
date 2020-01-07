const tf = require('@tensorflow/tfjs-node')

const csvtojson = require('csvtojson')

async function createData(filename) {
    let rawData = await csvtojson().fromFile(filename)

    let xs = []
    let ys = []
    let xv = []
    let yv = []

    rawData.map((value, index) => {
        let result = []
        let tempStr = rawData[index]['Features (X)'].replace('[', '').replace(']', '')
        let temp = tempStr.split(',')
        temp.map(item => {
            result.push(Number.parseFloat(item))
        })
        xs.push(result)
        ys.push(Number.parseFloat(rawData[index]['Label (Y)']))
    })

    return [xs, ys]
}

function callback(epoch, logs) {
    console.log(`ep:${epoch}\tlogs:${JSON.stringify(logs)}`)
}

async function trainModel(inputs, outputs, trainingsize, window_size, n_epochs, learning_rate, n_layers, callback) {

    const input_layer_shape = window_size;
    const input_layer_neurons = 100

    const rnn_input_layer_features = 10;
    const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;

    const rnn_input_shape = [rnn_input_layer_features, rnn_input_layer_timesteps];
    const rnn_output_neurons = 20;

    const rnn_batch_size = window_size;

    const output_layer_shape = rnn_output_neurons;
    const output_layer_neurons = 1;

    const model = tf.sequential();

    let X = inputs.slice(0, Math.floor(trainingsize / 100 * inputs.length));
    let Y = outputs.slice(0, Math.floor(trainingsize / 100 * outputs.length));

    const xs = tf.tensor2d(X, [X.length, X[0].length]).div(tf.scalar(10));
    const ys = tf.tensor2d(Y, [Y.length, 1]).reshape([Y.length, 1]).div(tf.scalar(10));

    model.add(tf.layers.dense({ units: input_layer_neurons, inputShape: [input_layer_shape] }));
    model.add(tf.layers.reshape({ targetShape: rnn_input_shape }));

    let lstm_cells = [];
    for (let index = 0; index < n_layers; index++) {
        lstm_cells.push(tf.layers.lstmCell({ units: rnn_output_neurons }));
    }

    model.add(tf.layers.rnn({
        cell: lstm_cells,
        inputShape: rnn_input_shape,
        returnSequences: false
    }));

    model.add(tf.layers.dense({ units: output_layer_neurons, inputShape: [output_layer_shape] }));

    model.compile({
        optimizer: tf.train.adam(learning_rate),
        loss: 'meanSquaredError'
    });

    const hist = await model.fit(xs, ys,
        {
            batchSize: rnn_batch_size, epochs: n_epochs, callbacks: {
                onEpochEnd: async (epoch, log) => {
                    callback(epoch, log);
                }
            }
        });

    return { model: model, stats: hist };
}

async function predictModel(model, xv) {
    let xv_tensor = tf.tensor2d(xv)
    let y_pred = model.predict(xv_tensor).mul(10);
    return Array.from(y_pred.dataSync());
}

(async () => {
    let training_size = 10
    let window_size = 50
    let n_epochs = 50
    let learning_rate = 0.01
    let n_layers = 4

    let [xs, ys] = await createData("tfjs-stocks-data.csv")

    let { model, hist } = await trainModel(xs, ys, training_size, window_size, n_epochs, learning_rate, n_layers, callback)

    let y_pred = await predictModel(model, xs.slice(0,2))
    console.log(y_pred)
})()