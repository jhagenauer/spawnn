package spawnn.ng.utils;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
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

import javax.imageio.ImageIO;

import org.jdom.Document;
import org.jdom.Element;
import org.jdom.input.SAXBuilder;
import org.jdom.output.Format;
import org.jdom.output.XMLOutputter;

import spawnn.ng.Connection;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.ColorBrewerUtil;
import spawnn.utils.ColorBrewerUtil.ColorMode;

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

	public static Map<double[],Set<double[]>> getBmuMapping( List<double[]> samples, List<double[]> neurons, Sorter<double[]> sorter ) {
		Map<double[],Set<double[]>> bmus = new HashMap<double[],Set<double[]>>();
		for( double[] d : neurons )
			bmus.put( d, new HashSet<double[]>() );
		for( double[] x : samples )
			bmus.get(sorter.getBMU(x, neurons)).add(x);
		return bmus;
	}
	
	public static void geoDrawNG(String fn, Map<double[],Double> neurons, Collection<Connection> conections, int[] ga, List<double[]> samples ) {
		int xScale = 1000;
		int yScale = 800;
		
		BufferedImage bufImg = new BufferedImage(xScale, yScale, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g2.fillRect(0, 0, xScale, yScale);

		// for scaling
		double maxX = Double.MIN_VALUE;
		double minX = Double.MAX_VALUE;
		double maxY = Double.MIN_VALUE;
		double minY = Double.MAX_VALUE;

		for (double[] n : samples ) {
			double x = n[ga[0]];
			double y = n[ga[1]];
			
			if (x > maxX)
				maxX = x;
			if (y > maxY)
				maxY = y;
			if (x < minX)
				minX = x;
			if (y < minY)
				minY = y;
		}
		
		for( double[] n : samples ) {
			g2.setColor(Color.GRAY);
			int x1 = (int)(xScale * (n[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (n[ga[1]] - minY)/(maxY-minY));
			g2.fillOval( x1 - 3, y1 - 3, 6, 6	);
		}

		if( conections != null )
		for( Connection c : conections ) {
			g2.setColor(Color.BLACK);
			double[] a = c.getA();
			double[] b = c.getB();
			int x1 = (int)(xScale * (a[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (a[ga[1]] - minY)/(maxY-minY));
			int x2 = (int)(xScale * (b[ga[0]] - minX)/(maxX-minX));
			int y2 = (int)(yScale * (b[ga[1]] - minY)/(maxY-minY));
			g2.drawLine(x1,y1,x2,y2);
		}

		Map<double[],Color> cMap = ColorBrewerUtil.valuesToColors(neurons, ColorMode.Blues);
		for( double[] n : neurons.keySet() ) {
			int x1 = (int)(xScale * (n[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (n[ga[1]] - minY)/(maxY-minY));
			
			g2.setColor(Color.BLACK);
			g2.fillOval( x1 - 8, y1 - 8, 16, 16	);
			
			g2.setColor(cMap.get(n));
			g2.fillOval( x1 - 7, y1 - 7, 14, 14	);
			
			/*g2.setColor(Color.BLACK);
			g2.drawString(""+n.hashCode(),x1,y1);*/
		}
		g2.dispose();

		try {
			ImageIO.write(bufImg, "png", new FileOutputStream(fn));
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}
}
