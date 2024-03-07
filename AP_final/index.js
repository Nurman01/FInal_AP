// Define CLASSES dictionary with name and type for each class
const CLASSES = {
  0: { name: 'Apple', type: 'fruit' },
  1: { name: 'Bitter Melon', type: 'vegetable' },
  2: { name: 'Brinjal Dotted', type: 'vegetable' },
  3: { name: 'Chilli', type: 'vegetable' },
  4: { name: 'Fig', type: 'fruit' },
  5: { name: 'Green Orange', type: 'fruit' },
  6: { name: 'Green Paper', type: 'vegetable' },
  7: { name: 'Khira', type: 'vegetable' },
  8: { name: 'Kiwi', type: 'fruit' },
  9: { name: 'Onion', type: 'vegetable' },
  10: { name: 'Pepper', type: 'vegetable' },
  11: { name: 'Pomogranate', type: 'fruit' },
  12: { name: 'Red Cabbage', type: 'vegetable' },
  13: { name: 'Sapodilla', type: 'fruit' },
  14: { name: 'SMG', type: 'fruit' },
  15: { name: 'Sponge Gourd', type: 'vegetable' },
  16: { name: 'Strawberry', type: 'fruit' },
  17: { name: 'Tomato Green', type: 'vegetable' },
  18: { name: 'Tomato Red', type: 'vegetable' },
  19: { name: 'Watermelon', type: 'fruit' }
};

// Define CALORIE_INFO dictionary with calorie value for each item
const CALORIE_INFO = {
  'Apple': 89,
  'Bitter Melon': 34,
  'Brinjal Dotted': 25,
  'Chilli': 40,
  'Fig': 74,
  'Green Orange': 43,
  'Green Paper': 20,
  'Khira': 16,
  'Kiwi': 61,
  'Onion': 40,
  'Pepper': 40,
  'Pomogranate': 83,
  'Red Cabbage': 25,
  'Sapodilla': 83,
  'SMG': 83,
  'Sponge Gourd': 15,
  'Strawberry': 32,
  'Tomato Green': 23,
  'Tomato Red': 18,
  'Watermelon': 30
};

const MODEL_PATH = 'model.json';

const IMAGE_SIZE = 150;
const TOPK_PREDICTIONS = 1; // Only keep the top prediction

let my_model;
const demo = async () => {
  status('Loading model...');

  my_model = await tf.loadLayersModel(MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  my_model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  // Make a prediction through the locally hosted cat.jpg.
  const catElement = document.getElementById('cat');
  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
    catElement.style.display = '';
  } else {
    catElement.onload = () => {
      predict(catElement);
      catElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};

/**
 * Given an image element, makes a prediction through my_model returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  status('Predicting...');

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in addition to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    // const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    // const normalized = img.sub(offset).div(offset);
    const normalized = img.div(255.0);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through my_model.
    return my_model.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
    `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from my_model.
 * @param topK The number of top predictions to show.
 */
async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({ value: values[i], index: i });
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    const classIndex = topkIndices[i];
    const className = CLASSES[classIndex].name;
    const classType = CLASSES[classIndex].type;
    const probability = topkValues[i];
    topClassesAndProbs.push({ className, classType, probability });
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {
  const predictionsElement = document.getElementById('predictions');

  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const topPrediction = classes[0];
  const predictedClass = topPrediction.className;
  const predictedType = topPrediction.classType;
  const calorieValue = CALORIE_INFO[predictedClass];

  // Construct the result text with escape characters for newline
  const resultText = `Name: ${predictedClass}\nCalories: ${calorieValue}\nType: ${predictedType}`;

  const resultElement = document.createElement('div');
  resultElement.className = 'result';
  resultElement.innerText = resultText;

  predictionContainer.appendChild(resultElement);

  predictionsElement.appendChild(predictionContainer);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

demo();
