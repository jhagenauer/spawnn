package spawnn.gui;

import java.awt.Color;
import java.io.File;
import java.util.Map;

import javax.swing.JPanel;

import spawnn.som.grid.GridPos;

public abstract class NeuronVisPanel <T> extends JPanel {

	private static final long serialVersionUID = 9155675342279074510L;
	public static int SELECTED_WIDTH = 3;
	public static int SELECTED_OPACITY = 64;
	
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
	
	public abstract void setColors(final Map<T, Color> colorMap, final Map<T,Color> selectedMap, final Map<T,Double> neuronValues);

	public abstract void saveImage(File fn, String mode);
}
