package spawnn.ng.utils;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.jdom.Document;
import org.jdom.Element;
import org.jdom.input.SAXBuilder;
import org.jdom.output.Format;
import org.jdom.output.XMLOutputter;

import spawnn.ng.sorter.Sorter;

public class NGUtils {
	
	//TODO expert and import should work on NG, maybe include context models and model parameters! 
	public static void saveGas( Collection<double[]> neurons, OutputStream os ) {
		int vDim = neurons.iterator().next().length;
		
		Element root = new Element("gas");		
		Element u = new Element("units");
		
		for( double[] vec : neurons  ) {
			Element p = new Element( "unit" );
			Element e = new Element("vector");
			
			for( int i = 0; i < vDim; i++ ) {
				Element v = new Element("value");
				v.setText(vec[i]+"");
				e.addContent(v);
			}
			p.addContent(e);
			u.addContent(p);
		}
		root.addContent(u);
				
		Document doc = new Document(root); 
		try {
		     XMLOutputter serializer = new XMLOutputter();
		     serializer.setFormat( Format.getPrettyFormat() );
		     serializer.output(doc, os);
		} catch (IOException e) {
		      System.err.println(e);
		}
	}
	
	public static Collection<double[]> loadGas( InputStream is ) {
		List<double[]> map = null;
		
		try {
			SAXBuilder builder = new SAXBuilder();
			Document doc = builder.build( is );  
			Element root = doc.getRootElement();
			map = new ArrayList<double[]>();
												
			for( Object o1 : root.getChild("units").getChildren() ) {
				Element e = (Element)o1;
								
				List<Double> v = new ArrayList<Double>();
				for( Object o2 : e.getChild("vector").getChildren() ) {
					Element e2 = (Element)o2;
					v.add( Double.parseDouble( e2.getText() ) );
				}
				
				double[] va = new double[v.size()];
				for( int i = 0; i < va.length; i++ )
					va[i] = v.get(i);
				
				map.add(va);
			}		
		} catch( Exception ex ) {
			ex.printStackTrace();
		}
		return map;
	}

	public static Map<double[],Set<double[]>> getBmuMapping( List<double[]> samples, List<double[]> neurons, Sorter<double[]> bg ) {
		Map<double[],Set<double[]>> bmus = new HashMap<double[],Set<double[]>>();
		for( double[] d : neurons )
			bmus.put( d, new HashSet<double[]>() );
		for( double[] x : samples ) {
			bg.sort(x, neurons );
			bmus.get(neurons.get(0)).add(x);
		}	
		return bmus;
	}
}
