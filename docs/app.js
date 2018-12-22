// load pre-trained model
let model;
tf.loadModel('./model/model.json')
    .then(pretrainedModel => {
        document.body.removeAttribute("style");
        document.getElementById("uk-input").setAttribute("placeholder", "Select file");
        model = pretrainedModel;
    })
    .catch(error => {
        console.log(error);
    });

let fileinfo;
let imgElement = document.getElementById('image-src');
let inputElement = document.getElementById('file-input');
inputElement.addEventListener('change', (e) => {
    fileinfo = e.target.files[0];
    imgElement.src = URL.createObjectURL(fileinfo);

    document.getElementById("inference").removeAttribute("disabled");
    document.getElementById("inference").removeAttribute("style");
}, false);


function binarize() {
    console.log(fileinfo);
    let mat = cv.imread(imgElement);
    var dst = new cv.Mat();
    let kernel = new cv.Mat.ones(new cv.Size(1, 1), cv.CV_8U);
    let rgbaPlanes = new cv.MatVector();

    // binalize
    cv.split(mat, rgbaPlanes); // red                green
    cv.addWeighted(rgbaPlanes.get(2), 0.5, rgbaPlanes.get(1), 0.5, 0, dst);
    cv.adaptiveThreshold(dst, dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 9, 10);
    cv.morphologyEx(dst, dst, cv.MORPH_OPEN, kernel);
    cv.resize(dst, dst, new cv.Size(400, 400));

    cv.imshow('canvas-output', dst);

    mat.delete();
    dst.delete();
    kernel.delete();
    rgbaPlanes.delete();
}


function getImageData() {
    const drawElement = document.getElementById('canvas-output');
    const inputWidth = inputHeight = 400;
    let imageData = drawElement.getContext('2d').getImageData(0, 0, inputWidth, inputHeight);
    return imageData;
}


function getAccuracyScores(imageData) {
    const score = tf.tidy(() => {
        // convert to tensor (shape: [width, height, channels])
        const channels = 1; // grayscale
        let input = tf.fromPixels(imageData, channels);
        // normalized
        input = tf.cast(input, 'float32');
        // reshape input format (shape: [batch_size, width, height, channels])
        input = input.expandDims();
        // predict
        return model.predict(input).dataSync();
    });
    return score;
}

function inference() {
    document.getElementById("reset").removeAttribute("disabled");
    document.getElementById("reset").removeAttribute("style");

    binarize();
    const imageData = getImageData();
    const accuracyScores = getAccuracyScores(imageData);
    console.log(accuracyScores);
    const maxAccuracy = accuracyScores.indexOf(Math.max.apply(null, accuracyScores));

    const li = {0: 'FBMessanger', 1: 'Instagram', 2: 'Invalid', 3: 'LINE', 4: 'Twitter'};
    console.log(li[maxAccuracy]);

    let c = 0;
    const elements = document.querySelectorAll(".accuracy");
    elements.forEach(el => {
        el.setAttribute("value", String(accuracyScores[c]));
        c++;
    });

    document.getElementById("inference").setAttribute("disabled", "");
    document.getElementById("inference").setAttribute("style", "cursor:not-allowed");
}

function reset_val() {
    URL.revokeObjectURL(fileinfo);

    const drawElement = document.getElementById('canvas-output');
    const context = drawElement.getContext('2d');
    context.clearRect(0, 0, drawElement.width, drawElement.height);
    let c = 0;
    const elements = document.querySelectorAll(".accuracy");
    elements.forEach(el => {
        el.setAttribute("value", String(0.0));
        c++;
    });
    document.getElementById("image-src").removeAttribute("src");

    document.getElementById("reset").setAttribute("disabled", "");
    document.getElementById("reset").setAttribute("style", "cursor:not-allowed");
}
