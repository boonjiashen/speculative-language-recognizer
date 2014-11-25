package speculative;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class Parser {

    /*
     * Parses the input XML and identifies sentences and cue phrases
     * Generates binary labeled parse trees using Stanford parser
     * as input to Stanford Sentiment Analyser
     */
    public static void main(String[] args) throws Exception {

        String inFile = "res/abstracts.xml";
        String outFile = "res/parsed.txt";

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
        // Parse the XML and add the processed data to sentences
        List<Sentence> sentences = parseXML(inFile);

        // Parse the sentences using the Stanford parser
        // Generate binary trees (required for Stanford sentiment analyser)
        // Label the trees
        List<String> labeled = BinarizeDataset.binarizeAndLabel(sentences.subList(0, 100));

        // Write out labeled trees
        PrintWriter writer = new PrintWriter(outFile, "UTF-8");
        for (String s : labeled) {
            writer.write(s);
            writer.println();
        }
        writer.close();
    }

    /*
     * Parses the XML file <filename>
     * Returns the sentences and their cues as a list of Sentence objects
     */
    public static List<Sentence> parseXML(String filename) throws Exception {

        File file = new File(filename);
        final List<Sentence> sentences = new ArrayList<Sentence>();

        SAXParserFactory factory = SAXParserFactory.newInstance();
        SAXParser saxParser = factory.newSAXParser();
        DefaultHandler handler = new DefaultHandler() {

            // Data structures for current sentence
            List<String> wordList = null;
            List<Integer[]> spec = null;
            boolean inSentence = false;
            boolean inCue = false;

            @Override
            public void startElement(String uri, String localName,String qName, 
                    Attributes attributes) throws SAXException {
                if (qName.equalsIgnoreCase("SENTENCE")) {
                    // Parsing sentence
                    if (inSentence)
                        throw new IllegalStateException("Nested sentences are not allowed");
                    inSentence = true;
                    wordList = new ArrayList<String>();
                    spec = new ArrayList<Integer[]>();
                } else if (qName.equalsIgnoreCase("CUE")) {
                    if (!inSentence)
                        throw new IllegalStateException("Cues must be inside sentences");
                    String type = attributes.getValue("type");
                    if (type != null && type.equalsIgnoreCase("SPECULATION")) {
                        // Parsing speculative cue
                        if (inCue)
                            throw new IllegalStateException("Nested cues are not allowed");
                        inCue = true;
                    }
                }
            }

            @Override
            public void endElement(String uri, String localName,
                    String qName) throws SAXException {
                if (qName.equalsIgnoreCase("SENTENCE")) {
                    // End of sentence, generate sentence object
                    inSentence = false;
                    if (wordList.isEmpty())
                        return;
                    String[] words = new String[wordList.size()];
                    for (int i = 0; i < wordList.size(); i++) {
                        words[i] = wordList.get(i);
                    }
                    // Add sentence and its the list of its speculative cues to 'sentences'
                    // Label 1 speculative, 0 for non-speculative
                    String label = (spec.isEmpty()) ? "0" : "1";
                    sentences.add(new Sentence(words, spec, label, "0"));
                } else if (qName.equalsIgnoreCase("CUE")) {
                    // End of cue phrase
                    inCue = false;
                }
            }

            @Override
            public void characters(char ch[], int start, int length) throws SAXException {
                if (inSentence) {
                    // Tokenize phrase and add to word list
                    // Note indices if cue phrase
                    String part = String.valueOf(ch, start, length);
                    String[] tokens = part.replaceAll("[^a-zA-Z ]", " ").trim().split("\\s+");
                    if (tokens != null && tokens[0].isEmpty())
                        return;
                    if (inCue) {
                        Integer[] arr = new Integer[2];
                        arr[0] = wordList.size();
                        arr[1] = wordList.size() + tokens.length - 1;
                        spec.add(arr);
                    }
                    for (int i = 0; i < tokens.length; i++) {
                        wordList.add(tokens[i]);
                    }
                }
            }

        };
        saxParser.parse(file, handler);
        return sentences;
    }
}
