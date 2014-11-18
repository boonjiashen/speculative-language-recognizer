package speculative;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class XMLParser {

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
        parseXML(filename, sentences);
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
     * Parses the XML file <filename>
     * Adds the sentences and their cues to 'sentences'
     */
    private static void parseXML(String filename, List<Sentence> sentences) throws Exception {
        File file = new File(filename);
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        DocumentBuilder db = dbf.newDocumentBuilder();
        Document doc = db.parse(file);
        doc.getDocumentElement().normalize();
        NodeList nodeLst = doc.getElementsByTagName("sentence");
        // Regex to replace non-alphabets in text
        String regex = "[^a-zA-Z ]";
        for (int s = 0; s < nodeLst.getLength(); s++) {
            // for each sentence
            Element sent = (Element) nodeLst.item(s);
            String sentenceWords = sent.getTextContent();
            // replace non-alphabets in sentence
            String[] words = sentenceWords.replaceAll(regex," ").split("\\s+");
            // list of speculate word start/end index pairs
            List<Integer[]> spec = new ArrayList<Integer[]>();
            // pointer to location in sentence
            int index = 0;

            NodeList cueNodes = sent.getElementsByTagName("cue");
            for (int c = 0; c < cueNodes.getLength(); c++) {
                // for each negation/speculation cue phrase
                // cue phrase start/end index pair
                Integer[] arr = new Integer[2];
                Element cueElmt = (Element) cueNodes.item(c);
                if (cueElmt.getAttribute("type").equals("speculation")) {
                    // consider only speculative cues
                    String cues = cueNodes.item(c).getChildNodes().item(0).getNodeValue();
                    // replace non-alphabets in cue phrase
                    String[] cueWords = cues.replaceAll(regex," ").split("\\s+");
                    for (int i = index; i < words.length; i++) {
                        if (words[i].equals(cueWords[0])) {
                            int start = i;
                            boolean isCue = true;
                            for (int cueNum = 1; cueNum < cueWords.length; cueNum++) {
                                if (!words[++i].equals(cueWords[cueNum])) {
                                    isCue = false;
                                }
                            }
                            if (isCue) {
                                // speculative cue matched in sentence, save indices
                                // and check next speculative cue if any
                                arr[0] = start;
                                arr[1] = start + cueWords.length - 1;
                                index = i + 1;
                                break;
                            }
                        }
                    }
                    spec.add(arr);
                }
            }
            // add sentence and its the list of its speculative cues to 'sentences'
            sentences.add(new Sentence(words, spec));
        }
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

    /*
     * Represents a sentence as a string array and a list of cue start/end index pairs
     */
    public static class Sentence {

        public String[] words = null;
        public List<Integer[]> spec = null;

        public Sentence(String[] words, List<Integer[]> spec) {
            this.words = words;
            this.spec = spec;
        }
    }
}
