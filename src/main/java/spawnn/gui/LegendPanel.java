package spawnn.gui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class LegendPanel<T> extends NeuronVisPanel<T> {

	private static final long serialVersionUID = -3421819319861456797L;
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
	public void saveImage(File fn, String mode) {
		// TODO Auto-generated method stub
	}
	
	//@Override
	public void paintComponent( Graphics g ) {
		if( clusterLegend )
			drawClusterLegend((Graphics2D) g);
		else
			drawContinoutLegend((Graphics2D) g);
	}
	
	public void drawContinoutLegend(Graphics2D g) {
		g.clearRect(0, 0, getWidth(), getHeight());
		
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
						
			if ( neurons.get(0) == t || neurons.get(neurons.size()-1) == t ) {
				g.setColor(Color.BLACK);
				g.drawLine((int) Math.round( x + 0.5 * cellWidth), cellHeight, (int) Math.round(x + 0.5 * cellWidth), (int) Math.round(cellHeight + cellHeight/4.0) );
												
				String s = df.format(neuronValues.get(t));
				int sWidth = g.getFontMetrics().stringWidth(s);
				g.drawString( s, (int)Math.round(x - 0.5 * sWidth + 0.5 * cellWidth ), (int) (Math.round(cellHeight + cellHeight/4.0 + 12)));
			}
			if( selectedMap.containsKey(t) ) {
				g.setColor( selectedMap.get(t) );
				g.drawLine((int) Math.round( x + 0.5 * cellWidth), 0, (int) Math.round(x + 0.5 * cellWidth), (int) Math.round(cellHeight + cellHeight/4.0) );
			}
			x += cellWidth;
		}			
		g.setColor(Color.BLACK);
		g.drawRect(xMargin, 0, (int)Math.round(width), cellHeight);
	}
	
	public void drawClusterLegend(Graphics2D g) {
		g.clearRect(0, 0, getWidth(), getHeight());
		
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
