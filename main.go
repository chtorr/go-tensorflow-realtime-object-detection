package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"image"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
	"github.com/faiface/pixel/text"
	"gocv.io/x/gocv"
	"golang.org/x/image/colornames"
	"golang.org/x/image/font/basicfont"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"

	_ "image/jpeg"
)

const (
	defaultModel         = "ssd_mobilenet_v1_coco_11_06_2017"
	defaultModelFilename = "frozen_inference_graph.pb"
	W                    = 640
	H                    = 480
)

var (
	tfSession *tf.Session
	tfGraph   *tf.Graph
	labels    []string

	cam   *gocv.VideoCapture
	frame gocv.Mat
)

func run() {
	cfg := pixelgl.WindowConfig{
		Title:  "Pixel Rocks!",
		Bounds: pixel.R(0, 0, W, H),
		VSync:  true,
	}
	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		panic(err)
	}

	cam, err := gocv.VideoCaptureDevice(0)
	if err != nil {
		log.Fatal("failed reading cam")
	}
	defer cam.Close()

	frame := gocv.NewMat()
	defer frame.Close()

	atlas := text.NewAtlas(basicfont.Face7x13, text.ASCII)

	mat := pixel.IM
	mat = mat.Moved(win.Bounds().Center())

	imd := imdraw.New(nil)

	var (
		frames = 0
		second = time.Tick(time.Second)
	)

	for !win.Closed() {
		if ok := cam.Read(frame); !ok {
			log.Fatal("failed reading cam")
		}

		gocv.Resize(frame, frame, image.Point{X: W, Y: H}, 0.0, 0.0, gocv.InterpolationNearestNeighbor)

		buf, _ := gocv.IMEncode(".jpg", frame)

		tensor, img, err := makeTensorFromImage(buf)
		if err != nil {
			log.Fatalf("error making input tensor: %v", err)
		}

		probabilities, classes, boxes, err := getObjectBoxes(tensor)
		if err != nil {
			log.Fatalf("error making prediction: %v", err)
		}

		// turn our video frame into a a sprite
		pic := pixel.PictureDataFromImage(img)
		sprite := pixel.NewSprite(pic, pic.Bounds())

		imd.Clear()
		win.Clear(colornames.Skyblue)
		sprite.Draw(win, mat)

		// Draw a box around the objects
		curObj := 0

		for probabilities[curObj] > 0.4 {
			x1 := pic.Bounds().Max.X * float64(boxes[curObj][1])
			x2 := pic.Bounds().Max.X * float64(boxes[curObj][3])
			y1 := pic.Bounds().Max.Y - (pic.Bounds().Max.Y * float64(boxes[curObj][0]))
			y2 := pic.Bounds().Max.Y - (pic.Bounds().Max.Y * float64(boxes[curObj][2]))

			txt := text.New(pixel.V(x1, y1), atlas)
			txt.Color = colornames.Blueviolet
			txt.WriteString(getLabel(curObj, probabilities, classes))
			txt.Draw(win, pixel.IM.Scaled(txt.Orig, 1))

			imd.Color = colornames.Blueviolet
			imd.Push(pixel.V(x1, y1), pixel.V(x2, y2))
			imd.Rectangle(1.0)

			curObj++
		}

		imd.Draw(win)
		win.Update()

		frames++
		select {
		case <-second:
			win.SetTitle(fmt.Sprintf("%s | FPS: %d", cfg.Title, frames))
			frames = 0
		default:
		}
	}
}

func main() {
	modelName := flag.String("model", defaultModel, "The name of the model to use from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md")
	modelDir := flag.String("dir", "", "Directory containing the trained model files. The directory will be created and the model downloaded into it if necessary")
	flag.Parse()
	if *modelDir == "" {
		flag.Usage()
		return
	}

	labelsFile := filepath.Join(*modelDir, "data/coco_labels.txt")
	modelFile := filepath.Join(*modelDir, *modelName, "frozen_inference_graph.pb")

	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		log.Fatal(err)
	}
	loadLabels(labelsFile)

	// Construct an in-memory graph from the serialized form.
	tfGraph = tf.NewGraph()
	if err := tfGraph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	tfSession, err = tf.NewSession(tfGraph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer tfSession.Close()

	pixelgl.Run(run)
}

func getLabel(idx int, probabilities []float32, classes []float32) string {
	index := int(classes[idx])
	label := fmt.Sprintf("%s (%2.0f%%)", labels[index], probabilities[idx]*100.0)

	return label
}

func loadLabels(labelsFile string) {
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
}

func decodeJpegGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

// TENSOR UTILITY FUNCTIONS
func makeTensorFromImage(img []byte) (*tf.Tensor, image.Image, error) {

	r := bytes.NewReader(img)

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(img))
	if err != nil {
		return nil, nil, err
	}
	// Creates a tensorflow graph to decode the jpeg image
	graph, input, output, err := decodeJpegGraph()
	if err != nil {
		return nil, nil, err
	}
	// Execute that graph to decode this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, nil, err
	}

	i, _, err := image.Decode(r)
	if err != nil {
		return nil, nil, err
	}
	return normalized[0], i, nil
}

func getObjectBoxes(input *tf.Tensor) (probabilities, classes []float32, boxes [][]float32, err error) {
	// Get all the input and output operations
	inputop := tfGraph.Operation("image_tensor")
	// Output ops
	o1 := tfGraph.Operation("detection_boxes")
	o2 := tfGraph.Operation("detection_scores")
	o3 := tfGraph.Operation("detection_classes")
	o4 := tfGraph.Operation("num_detections")

	output, err := tfSession.Run(
		map[tf.Output]*tf.Tensor{
			inputop.Output(0): input,
		},
		[]tf.Output{
			o1.Output(0),
			o2.Output(0),
			o3.Output(0),
			o4.Output(0),
		},
		nil)
	if err != nil {
		log.Fatalf("Error running session: %v", err)
	}

	probabilities = output[1].Value().([][]float32)[0]
	classes = output[2].Value().([][]float32)[0]
	boxes = output[0].Value().([][][]float32)[0]

	return
}
