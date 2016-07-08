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
	Map<T, Color> colorMap;
	Map<T, Double> neuronValues;
	
	public LegendPanel() {
		Dimension dim = new Dimension(400, 36);
		setPreferredSize(dim);
		setSize(dim);
		setMinimumSize(dim);
	}

	@Override
	public void setColors(Map<T, Color> colorMap, Map<T, Color> selectedMap, Map<T, Double> neuronValues) {
		this.colorMap = colorMap;
		this.neuronValues = neuronValues;
		repaint();
	}

	@Override
	public void saveImage(File fn, String mode) {
		// TODO Auto-generated method stub
	}
	
	@Override
	public void paintComponent( Graphics g1 ) {
		Graphics2D g = (Graphics2D)g1;
				
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
		for( T t : neurons ) {
			g.setColor(colorMap.get(t));
			g.fillRect( (int)Math.round(x), 0, (int)Math.round(cellWidth), cellHeight);
						
			if (neurons.get(0) == t || neurons.get(neurons.size()-1) == t ) {
				g.setColor(Color.BLACK);
				g.drawLine((int) Math.round( x + 0.5 * cellWidth), cellHeight, (int) Math.round(x + 0.5 * cellWidth), (int) Math.round(cellHeight + cellHeight/4.0) );
				
				String s = df.format(neuronValues.get(t));
				int sWidth = g.getFontMetrics().stringWidth(s);
				g.drawString( s, (int)Math.round(x - 0.5 * sWidth + 0.5 * cellWidth ), (int) (Math.round(cellHeight + cellHeight/4.0 + 12)));
			}		
			x += cellWidth;
		}			
		
		g.setColor(Color.BLACK);
		g.drawRect(xMargin, 0, (int)Math.round(width), cellHeight);
	}
}
