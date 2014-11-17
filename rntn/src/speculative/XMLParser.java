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

 public static void main(String argv[]) {

  try {
  File file = new File("C:\\Users\\Rasiga\\Downloads\\bioscope\\abstracts.xml");
  DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
  DocumentBuilder db = dbf.newDocumentBuilder();
  Document doc = db.parse(file);
  doc.getDocumentElement().normalize();
  System.out.println("Root element " + doc.getDocumentElement().getNodeName());
  NodeList nodeLst = doc.getElementsByTagName("sentence");
  System.out.println("Number of sentences:"+nodeLst.getLength());
  
  for(int s=0;s<nodeLst.getLength();s++){ // for each sentence
	  
	  Element sent = (Element)nodeLst.item(s);
	  System.out.println("Sentence"+s+":"+sent.getTextContent());
	  String sentenceWords = sent.getTextContent();
	  System.out.println("Num of parts in sentence:"+sent.getChildNodes().getLength());
	  String[] words = sentenceWords.replaceAll("[^a-zA-Z ]","").split("\\s+");
	  List<Integer[]> spec = new ArrayList<Integer[]>(); // list of speculative word pairs
	  int index = 0;
	  
		  NodeList cueNode = sent.getElementsByTagName("cue");
		  for(int c=0;c<cueNode.getLength();c++){
			  Integer[] arr = new Integer[10];
			  int j=0;
			  Element cueElmt = (Element)cueNode.item(c);
			  if(cueElmt.getAttribute("type").equals("speculation")){
				  System.out.println("Cue words:");
				  System.out.println(cueNode.item(c).getChildNodes().item(0).getNodeValue());		
				  String cues = cueNode.item(c).getChildNodes().item(0).getNodeValue();
				  String[] cueWords = cues.replaceAll("[^a-zA-Z ]","").split("\\s+");
				  for(int cueNum = 0;cueNum<cueWords.length;cueNum++){
					  for(int i = index; i<words.length;i++){
						  if(words[i].equals(cueWords[cueNum])){
							  arr[j++] = i;						  
							  index = i+1;
							  break;
						  }
					  }					  
				  }
				  spec.add(arr);
			  }
	  }
	  /*for(int n = 0;n<spec.size();n++){
		  Integer[] pair = spec.get(n);
		  for(int num=0;pair[num]!=null;num++)
			  System.out.print(pair[num]+",");
		  System.out.println();
	  }*/
  }
  }  
  catch (Exception e) {
    e.printStackTrace();
  }
 }
}
