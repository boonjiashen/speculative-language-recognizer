package speculative;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Pipeline {

    public static void main(String[] args) throws Exception {

        String filename = null;

        for (int argIndex = 0; argIndex < args.length; argIndex++) {
            if (args[argIndex].equalsIgnoreCase("-input")) {
                filename = args[++argIndex];
            } else {
                System.err.println("Unknown argument " + args[argIndex]);
                System.exit(2);
            }
        }

        if (filename == null) {
            filename = "res/abstracts.xml";
        }

        // List of sentences along with cue phrases (if any)
        List<Sentence> sentences = new ArrayList<Sentence>();
        // Parse the XML and add the processed data to sentences
        XMLParser.parseXML(filename, sentences);
        // Generate training/test data
        Collections.shuffle(sentences);
        int trainingSize = sentences.size()/2;
        int testSize = 500;
        List<Sentence> trainingData = sentences.subList(0, trainingSize);
        List<Sentence> testData = sentences.subList(trainingSize, trainingSize + testSize);
        // Generate the output in required format
        outputText(trainingData);
    }

    /*
     * Prints labeled sentences/cue phrases in format accepted by 
     * edu.stanford.nlp.sentiment.BuildBinarizedDataset.main
     */
    private static void outputText(List<Sentence> sentences) {
        for (int i = 0; i < sentences.size(); i++) {
            Sentence sentence = sentences.get(i);
            String[] words = sentence.words;
            List<Integer[]> spec = sentence.spec;
            boolean speculative = !spec.isEmpty();
            // Label 1 if speculative, 0 if not
            String label = speculative ? "1" : "0";
            System.out.print(label);
            for (String word : words) {
                System.out.print(" " + word);
            }
            System.out.println();
            if (speculative) {
                for (Integer[] arr : spec) {
                    System.out.print(label);
                    for (int j = arr[0]; j <= arr[1]; j++) {
                        System.out.print(" " + words[j]);
                    }
                    System.out.println();
                }
            }
            System.out.println();
        }
    }
}
