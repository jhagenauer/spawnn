package spawnn.gui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;
import javax.swing.JPanel;

import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;

import spawnn.som.grid.GridPos;

public abstract class NeuronVisPanel <T> extends JPanel {

	private static final long serialVersionUID = 9155675342279074510L;
	public static int SELECTED_WIDTH = 3;
	
	public void addNeuronSelectedListener(NeuronSelectedListener<T> listener) {
		listenerList.add(NeuronSelectedListener.class, listener);
	}

	public void removeNeuronSelectedListener(NeuronSelectedListener<T> listener) {
		listenerList.remove( NeuronSelectedListener.class, listener );
	}

	public void fireNeuronSelectedEvent(NeuronSelectedEvent evt) {
		for( NeuronSelectedListener<GridPos> listener : listenerList.getListeners( NeuronSelectedListener.class ) )
			listener.neuronSelectedOccured(evt);
	}	
	
	public abstract void setGridColors(final Map<T, Color> colorMap, final Map<T,Color> selectedMap, final Map<T,Double> neuronValues);

	public abstract void saveImage(File fn, String mode);
}
