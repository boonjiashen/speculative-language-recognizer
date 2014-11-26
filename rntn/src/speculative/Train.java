package speculative;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.stanford.nlp.sentiment.RNNOptions;
import edu.stanford.nlp.sentiment.RNNTrainOptions;
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
        String outFile = "res/model.txt";

        for (int argIndex = 0; argIndex < args.length; argIndex++) {
            if (args[argIndex].equalsIgnoreCase("-input")) {
                inFile = args[++argIndex];
            } else if (args[argIndex].equalsIgnoreCase("-output")) {
                outFile = args[++argIndex];
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
        //TODO: set the correct values for data size
        
        //int trainingSize = sentences.size()/2;
        //int testSize = Math.min(500, sentences.size() - trainingSize);
        //int devSize = ;
        
        int trainingSize = 10;
        int devSize = 5;
        int testSize = 5;
        List<String> trainingData = sentences.subList(0, trainingSize);
        List<String> devData = sentences.subList(trainingSize,trainingSize+devSize);
        List<String> testData = sentences.subList(trainingSize+devSize,trainingSize+devSize+testSize);
        
        //write the training, dev and test data to files
        PrintWriter pw = new PrintWriter("res\\train.txt","UTF-8");
        for(int i=0;i<trainingSize;i++)
        	pw.println(trainingData.get(i));
        pw.close();
        
        pw = new PrintWriter("res\\dev.txt","UTF-8");
        for(int i=0;i<devSize;i++)
        	pw.println(devData.get(i));
        pw.close();
        
        pw = new PrintWriter("res\\test.txt","UTF-8");
        for(int i=0;i<testSize;i++)
        	pw.println(testData.get(i));
        pw.close();
        
        // Train sentiment analysis model
        // TODO
        SentimentTraining st = new SentimentTraining();
        RNNOptions op = new RNNOptions();

        String trainPath = "C:\\Users\\Rasiga\\speculative\\speculative-language-recognizer\\rntn\\res\\train.txt";
        String devPath = "C:\\Users\\Rasiga\\speculative\\speculative-language-recognizer\\rntn\\res\\dev.txt";
        boolean runTraining = true;
        boolean filterUnknown = false;
        String modelPath = "C:\\Users\\Rasiga\\speculative\\speculative-language-recognizer\\rntn\\res\\toymodel.ser";

        // read in the trees
        List<Tree> trainingTrees = SentimentUtils.readTreesWithGoldLabels(trainPath);
        System.err.println("Read in " + trainingTrees.size() + " training trees");
        if (filterUnknown) {
          trainingTrees = SentimentUtils.filterUnknownRoots(trainingTrees);
          System.err.println("Filtered training trees: " + trainingTrees.size());
        }

        List<Tree> devTrees = null;
        if (devPath != null) {
          devTrees = SentimentUtils.readTreesWithGoldLabels(devPath);
          System.err.println("Read in " + devTrees.size() + " dev trees");
          if (filterUnknown) {
            devTrees = SentimentUtils.filterUnknownRoots(devTrees);
            System.err.println("Filtered dev trees: " + devTrees.size());
          }
        }
        //setting RNN options
        String[] options = new String[30];
        options[0] = "-numClasses";
        options[1] = "2";
        options[2] = "-simplifiedModel";
        options[3] = "-classNames";
        options[4] = "speculative,non-speculative";
        options[5] = "-equivalenceClasses";
        options[6] = " ";
        options[7] = "-equivalenceClassNames";
        options[8] = " ";
        
        int index = 0;
        while(options[index]!=null)
        	index = op.setOption(options,index);
        
        String[] trainOptions = new String[30];
        trainOptions[0] = "-batchSize";
        trainOptions[1] = "2";
        trainOptions[2] = "-epochs";
        trainOptions[3] = "10";
        trainOptions[4] = "-learningRate";
        trainOptions[5] = "0.01";
        
        //RNNTrainOptions trainOp = new RNNTrainOptions();
        
        index = 0;
        while(trainOptions[index]!=null)
        	index = op.trainOptions.setOption(trainOptions,index);
        
        System.err.println("Sentiment model options:\n" + op);
        SentimentModel model = new SentimentModel(op, trainingTrees);
        
          if (runTraining) {
            st.train(model, modelPath, trainingTrees, devTrees);
            model.saveSerialized(modelPath);
          }
    }

}
