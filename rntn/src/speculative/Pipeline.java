package speculative;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class Pipeline {

    public static void main(String[] args) throws Exception {

        String inFile = "res/abstracts.xml";
        String outFile = "res/train.txt";

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

        // List of sentences along with cue phrases (if any)
        List<Sentence> sentences = new ArrayList<Sentence>();
        // Parse the XML and add the processed data to sentences
        XMLParser.parseXML(inFile, sentences);
        // Generate training/test data
        // FIXME: Re-enable shuffling after testing
        // Collections.shuffle(sentences);
        int trainingSize = sentences.size()/2;
        int testSize = 500;
        List<Sentence> trainingData = sentences.subList(0, trainingSize);
        List<Sentence> testData = sentences.subList(trainingSize, trainingSize + testSize);

        // Parse the sentences using the Stanford parser
        // Generate binary trees (required for Stanford sentiment analyser)
        // Label the trees
        // FIXME: pass training data to BinarizeDataset.binarizeAndLabel
        List<String> labeled = BinarizeDataset.binarizeAndLabel(sentences.subList(0, 100));

        // Write out labeled trees
        PrintWriter writer = new PrintWriter(outFile, "UTF-8");
        for (String s : labeled) {
            writer.write(s);
            writer.println();
        }
        writer.close();
    }
}
