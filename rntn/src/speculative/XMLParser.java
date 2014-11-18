package speculative;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class XMLParser {

    /*
     * Parses the XML file <filename>
     * Adds the sentences and their cues to 'sentences'
     */
    public static void parseXML(String filename, List<Sentence> sentences) throws Exception {
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
}
