package spawnn.gui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;
import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;

public class LegendPanel<T> extends NeuronVisPanel<T> {

	private static final long serialVersionUID = -3421819319861456797L;
	private static Logger log = Logger.getLogger(LegendPanel.class);
	private Map<T, Color> colorMap;
	private Map<T, Double> neuronValues;
	private Map<T, Color> selectedMap;
	private boolean clusterLegend = false;;
	
	public LegendPanel() {
		Dimension dim = new Dimension(500, 36);
		setOpaque(true);
		setPreferredSize(dim);
		setSize(dim);
		setMinimumSize(dim);
	}
	
	public void setClusterLegend(boolean clusterLegend ) {
		this.clusterLegend = clusterLegend;
	}
	
	@Override
	public void setColors(Map<T, Color> colorMap, Map<T, Color> selectedMap, Map<T, Double> neuronValues) {
		this.colorMap = colorMap;
		this.neuronValues = neuronValues;
		this.selectedMap = selectedMap;
		repaint();
	}

	@Override
	public void saveImage(File fn, ImageMode mode) {
		try {		
			FileOutputStream stream = new FileOutputStream(fn);			
			int width = getWidth();
			int height = getHeight();
			if( mode == ImageMode.PNG ) {
				BufferedImage bufImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
				Graphics2D g = bufImage.createGraphics();

				g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
				draw(g);
				
				ImageIO.write(bufImage, "PNG", stream);
			} else if( mode == ImageMode.EPS ) {
				EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
				g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext());
				g.setupDocument(stream, width, height ); 	
								
				draw(g);
				g.finish();
			} else {
				log.debug("Unknown file format!");
			}
			stream.flush();
			stream.close();
		} catch( Exception e ) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void paintComponent( Graphics g ) {
		super.paintComponent(g);
		draw((Graphics2D) g);
	}
	
	public void draw( Graphics2D g ) {
		if( clusterLegend )
			drawClusterLegend(g);
		else
			drawContinoutLegend(g);
	}
	
	public void drawContinoutLegend(Graphics2D g) {
		//g.clearRect(0, 0, getWidth(), getHeight());
		
		List<T> neurons = new ArrayList<>(neuronValues.keySet());
		Collections.sort(neurons, new Comparator<T>() {
			@Override
			public int compare(T o1, T o2) {
				return Double.compare(neuronValues.get(o1), neuronValues.get(o2) );
			}
		});
				
		DecimalFormat df = new DecimalFormat("#0.00");
		int xMargin = 18;
		double width = getWidth() - 2 * xMargin;
		double cellWidth = width/neurons.size();
		int cellHeight = (int)Math.round( getHeight()/2.0 );
		
		double x = xMargin;
		for( int i = 0; i < neurons.size(); i++ ) {
			T t = neurons.get(i);
			g.setColor(colorMap.get(t));
			g.fillRect( (int)Math.round(x), 0, (int)Math.round(cellWidth), cellHeight);
						
			if (i == 0 || i == neurons.size()-1 || i == neurons.size()/2 ) {
				g.setColor(Color.BLACK);
				g.drawLine((int) Math.round( x + 0.5 * cellWidth), cellHeight, (int) Math.round(x + 0.5 * cellWidth), (int) Math.round(cellHeight + cellHeight/4.0) );
												
				String s = df.format(neuronValues.get(t));
				int sWidth = g.getFontMetrics().stringWidth(s);
				g.drawString( s, (int)Math.round(x - 0.5 * sWidth + 0.5 * cellWidth ), (int) (Math.round(cellHeight + cellHeight/4.0 + 12)));
			}
			/*if( selectedMap.containsKey(t) ) {
				g.setColor( selectedMap.get(t) );
				g.drawLine((int) Math.round( x + 0.5 * cellWidth), 0, (int) Math.round(x + 0.5 * cellWidth), (int) Math.round(cellHeight + cellHeight/4.0) );
			}*/
			x += cellWidth;
		}			
		g.setColor(Color.BLACK);
		g.drawRect(xMargin, 0, (int)Math.round(width), cellHeight);
	}
	
	public void drawClusterLegend(Graphics2D g) {
		//g.clearRect(0, 0, getWidth(), getHeight());
		
		List<T> neurons = new ArrayList<>(neuronValues.keySet());
		Collections.sort(neurons, new Comparator<T>() {
			@Override
			public int compare(T o1, T o2) {
				return Double.compare(neuronValues.get(o1), neuronValues.get(o2) );
			}
		});
				
		int xMargin = 18;
		double width = getWidth() - 2 * xMargin;
		double cellWidth = width/neurons.size();
		int cellHeight = (int)Math.round( getHeight()/2.0 );
		
		double x = xMargin;
		double lastX = x;
		for( int i = 0; i < neurons.size(); i++ ) {
			T t = neurons.get(i);
			Color nCol = null;
			if( i < neurons.size()-1 )
				nCol = colorMap.get(neurons.get(i+1));
			
			x += cellWidth;
			
			if( colorMap.get(t) != nCol || nCol == null ) {
				g.setColor(colorMap.get(t));
				g.fillRect( (int)Math.round( lastX ), 0, (int)Math.round( x-lastX + 1), cellHeight);
				
				int midX = (int)Math.round( lastX + 0.5 * (x - lastX));
				
				g.setColor(Color.BLACK);
				g.drawLine(midX, cellHeight, midX, (int) Math.round(cellHeight + cellHeight/4.0) );
													
				String s = neuronValues.get(t).intValue()+"";
				int sWidth = g.getFontMetrics().stringWidth(s);
				g.drawString( s, (int)Math.round(midX - 0.5 * sWidth), (int) (Math.round(cellHeight + cellHeight/4.0 + 12)));
				
				
				lastX = x;
			}
		}	
		
		
		x = xMargin;
		for( int i = 0; i < neurons.size(); i++ ) {
			T t = neurons.get(i);
			
			
			x += cellWidth;
		}						
		g.setColor(Color.BLACK);
		g.drawRect(xMargin, 0, (int)Math.round(width), cellHeight);
	}
}
