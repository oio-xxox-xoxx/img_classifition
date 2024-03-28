const tf = require("@tensorflow/tfjs");
const mobilenet = require("@tensorflow-models/mobilenet");
const tfnode = require("@tensorflow/tfjs-node");
const fs = require("fs");

const imageBuffer = fs.readFileSync("./rabbit.jpg");
const tfimage = tfnode.node.decodeImage(imageBuffer);

const mobilenetModel = await mobilenet.load();
const predictions = await mobilenetModel.classify(tfimage);
console.log("Classification Results:", predictions);
