package speculative;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.stanford.nlp.sentiment.RNNOptions;
import edu.stanford.nlp.sentiment.SentimentModel;
import edu.stanford.nlp.sentiment.SentimentTraining;
import edu.stanford.nlp.sentiment.SentimentUtils;
import edu.stanford.nlp.trees.Tree;

public class Train {

    /*
     * Loads parse trees generated by Parser
     * Splits data into training and test sets
     * Trains Sentiment Analysis model
     */ 
    public static void main(String[] args) throws Exception {

        String inFile = "res/parsed.txt";
        String trainPath = "res/train.txt";
        String devPath = "res/dev.txt";
        String testPath = "res/test.txt";
        String modelPath = "res/model.ser.gz";
        boolean log = true;
        String logFile = "res/log.txt";

        for (int argIndex = 0; argIndex < args.length; argIndex++) {
            if (args[argIndex].equalsIgnoreCase("-input")) {
                inFile = args[++argIndex];
            } else if (args[argIndex].equalsIgnoreCase("-train")) {
                trainPath = args[++argIndex];
            } else if (args[argIndex].equalsIgnoreCase("-dev")) {
                devPath = args[++argIndex];
            } else if (args[argIndex].equalsIgnoreCase("-test")) {
                testPath = args[++argIndex];
            } else if (args[argIndex].equalsIgnoreCase("-model")) {
                modelPath = args[++argIndex];
            } else if (args[argIndex].equalsIgnoreCase("-log")) {
                logFile = args[++argIndex];
            } else {
                System.err.println("Unknown argument " + args[argIndex]);
                System.exit(2);
            }
        }

        // Load parse trees generated by Parser
        List<String> sentences = new ArrayList<String>();
        FileReader file = new FileReader(inFile);
        BufferedReader binfile = new BufferedReader(file);
        String sentence = binfile.readLine();
        while (sentence != null) {
            sentences.add(sentence);
            sentence = binfile.readLine();
        }
        binfile.close();

        // Generate training/test data
        Collections.shuffle(sentences);

        int trainingSize = (int) (sentences.size() * 0.6);
        int testSize = (int) (sentences.size() * 0.3);
        int devSize = sentences.size() - trainingSize - testSize;

        List<String> trainingData = sentences.subList(0, trainingSize);
        List<String> devData = sentences.subList(trainingSize, trainingSize + devSize);
        List<String> testData = sentences.subList(trainingSize + devSize, 
                trainingSize + devSize + testSize);

        //write the training, dev and test data to files
        PrintWriter pw = new PrintWriter(trainPath,"UTF-8");
        for(int i = 0; i < trainingSize; i++)
            pw.println(trainingData.get(i));
        pw.close();

        pw = new PrintWriter(devPath,"UTF-8");
        for(int i = 0; i < devSize; i++)
            pw.println(devData.get(i));
        pw.close();

        pw = new PrintWriter(testPath,"UTF-8");
        for(int i = 0;i < testSize; i++)
            pw.println(testData.get(i));
        pw.close();

        // Train sentiment analysis model
        RNNOptions op = new RNNOptions();

        // read in the trees
        List<Tree> trainingTrees = SentimentUtils.readTreesWithGoldLabels(trainPath);
        System.err.println("Read in " + trainingTrees.size() + " training trees");

        List<Tree> devTrees = null;
        if (devPath != null) {
            devTrees = SentimentUtils.readTreesWithGoldLabels(devPath);
            System.err.println("Read in " + devTrees.size() + " dev trees");
        }

        //setting RNN options
        List<String> options = new ArrayList<String>();
        options.add("-binaryModel");
        options.add("-classNames");
        options.add("non-speculative,speculative");
        options.add("-equivalenceClassNames");
        options.add("non-speculative,speculative");
        String[] optionsArr = new String[options.size()];
        for (int i = 0; i < options.size(); i++) {
            optionsArr[i] = options.get(i);
        }
        int index = 0;
        while (index < optionsArr.length) {
            index = op.setOption(optionsArr,index);
        }

        List<String> trainOptions = new ArrayList<String>();
        trainOptions.add("-epochs");
        trainOptions.add("401");
        String[] trainOptionsArr = new String[trainOptions.size()];
        for (int i = 0; i < trainOptions.size(); i++) {
            trainOptionsArr[i] = trainOptions.get(i);
        }
        index = 0;
        while(index < trainOptionsArr.length) {
            index = op.trainOptions.setOption(trainOptionsArr, index);
        }

        System.out.println("Sentiment model options:\n" + op);
        SentimentModel model = new SentimentModel(op, trainingTrees);

        PrintStream console = System.out;
        if (log) {
        	System.out.println("Redirecting output to log file: " + logFile);
            File lfile = new File(logFile);
            FileOutputStream fos = new FileOutputStream(lfile);
            PrintStream ps = new PrintStream(fos);
            System.setOut(ps);
            System.setErr(ps);
        }
        SentimentTraining.train(model, modelPath, trainingTrees, devTrees);
        model.saveSerialized(modelPath);
        if (log) {
            System.setOut(console);
            System.setErr(console);
        }
        System.out.println("Training complete");
    }
}
