package spawnn.gui;

import java.awt.CardLayout;
import java.awt.Cursor;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Polygon;

import net.miginfocom.swing.MigLayout;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class DistMatrixDialog extends JDialog implements ActionListener {

	private static final long serialVersionUID = 2783445681242978907L;
	
	JButton ok, cancel, save, create, load;
	JPanel cards;
	JComboBox cb, adjCb;
	JTextField power, knns, selFile;
	JCheckBox rowNorm, adjIncIdent, knnIncIdent;
	File file = null;

	public static enum DistMatType {InvDistance, kNN, Adjacency};
	
	public static enum AdjMode {Queen,Rook};
	
	private enum Card { one, two, three };
	
	private Dist<double[]> dist;
	private SpatialDataFrame sdf;
	private List<double[]> samples;
	
	public DistMatrixDialog(Frame parent, String string, boolean b, List<double[]> samples, SpatialDataFrame sd, int[] ga ) {
		super(parent, string, b);
		
		this.dist = new EuclideanDist(ga);
		this.sdf = sd;
		this.samples = samples;
		
		boolean adjEnable = sd.geoms != null && ( sd.geoms.get(0) instanceof Polygon || sd.geoms.get(0) instanceof MultiPolygon );
				
		setLayout(new MigLayout(""));

		add(new JLabel("Distance:"));
		cb = new JComboBox();
		if( ga.length > 0 ) {
			cb.addItem(DistMatType.InvDistance);
			cb.addItem(DistMatType.kNN);
		}
		if( adjEnable )
			cb.addItem(DistMatType.Adjacency);
		cb.addActionListener(this);
		add(cb, "span 2, wrap");
						
		// inv dist
		JPanel invDistPanel = new JPanel(new MigLayout());
		invDistPanel.add(new JLabel("Power:"));
		power = new JTextField("1", 3);
		invDistPanel.add(power, "wrap");
		
		// knn
		JPanel knnPanel = new JPanel(new MigLayout());
		knnPanel.add(new JLabel("k:"));
		knns = new JTextField("5", 3);
		knnPanel.add(knns, "wrap");
		knnPanel.add( new JLabel("Include identity:"));
		knnIncIdent = new JCheckBox();
		knnIncIdent.setEnabled(true);
		knnPanel.add( knnIncIdent, "wrap" );
		
		// adj
		JPanel adjPanel = new JPanel(new MigLayout());
		adjPanel.add(new JLabel("Type:"));
		adjCb = new JComboBox();
		adjCb.setModel(new DefaultComboBoxModel(AdjMode.values()));
		adjPanel.add(adjCb, "wrap");
		adjPanel.add( new JLabel("Include identity:"));
		adjIncIdent = new JCheckBox();
		adjIncIdent.setEnabled(true);
		adjPanel.add( adjIncIdent, "wrap" );
		
		cards = new JPanel(new CardLayout());
		if( ga.length > 0 ) {
			cards.add(invDistPanel, Card.one.toString());
			cards.add(knnPanel, Card.two.toString());
		}
		if( adjEnable )
			cards.add(adjPanel, Card.three.toString());
		
		add(cards, "span 3, wrap");
				
		add( new JLabel("Row-normalize:"));
		rowNorm = new JCheckBox();
		rowNorm.setSelected(true);
		add( rowNorm, "wrap" );
		
		create = new JButton("Create");
		create.addActionListener(this);
		add(create, "span 2, split 3");
						
		save = new JButton("Save...");
		save.addActionListener(this);
		save.setEnabled(false);
		add(save, "");
		
		load = new JButton("Load...");
		load.addActionListener(this);
		add(load, "wrap");

		ok = new JButton("OK");
		ok.setEnabled(false);
		ok.addActionListener(this);
		cancel = new JButton("Cancel");
		cancel.addActionListener(this);
		add(ok, "span2, split 2, push");
		add(cancel);

		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		pack();
		setAlwaysOnTop(true);
		setLocationRelativeTo(parent);
		setVisible(true);
	}
	
	boolean okPressed = false;
	Map<double[], Map<double[], Double>> dMap = null;

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == cb) {
			CardLayout cl = (CardLayout) (cards.getLayout());
			if (cb.getSelectedItem() == DistMatType.InvDistance )
				cl.show(cards, Card.one.toString());
			else if( cb.getSelectedItem() == DistMatType.kNN )
				cl.show(cards, Card.two.toString());
			else
				cl.show(cards, Card.three.toString());
		} else if (e.getSource() == create) {
			setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
			
			if( cb.getSelectedItem() == DistMatType.Adjacency ) {
				dMap = GeoUtils.contiguityMapToDistanceMap( GeoUtils.getContiguityMap(samples, sdf.geoms, adjCb.getSelectedItem() == AdjMode.Rook, adjIncIdent.isSelected() ));
			} else if( cb.getSelectedItem() == DistMatType.InvDistance ) {
				dMap = GeoUtils.getInverseDistanceMatrix(samples, dist, Double.parseDouble(power.getText() ) );
			} else { // knn
				dMap = GeoUtils.listsToWeightsOld( GeoUtils.getKNNs(samples, dist, Integer.parseInt(knns.getText()), knnIncIdent.isSelected() ));
			}
			
			if( rowNorm.isSelected() )
				GeoUtils.rowNormalizeMatrix(dMap);
			
			setCursor(Cursor.getDefaultCursor());
			save.setEnabled(true);
			ok.setEnabled(true);
		} else if( e.getSource() == save ) {
			JFileChooser fc = new JFileChooser();
			if( fc.showSaveDialog(this) == JFileChooser.APPROVE_OPTION )
				  GeoUtils.writeDistMatrixKeyValue(dMap, samples, fc.getSelectedFile() );
		} else if( e.getSource() == load ) { 
			JFileChooser fc = new JFileChooser();
			if( fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION ) {
				try {
					setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
					dMap = GeoUtils.readDistMatrixKeyValue(samples, fc.getSelectedFile() );
					setCursor(Cursor.getDefaultCursor());
				} catch (NumberFormatException e1) {
					e1.printStackTrace();
				} catch (FileNotFoundException e1) {
					e1.printStackTrace();
				} catch (IOException e1) {
					e1.printStackTrace();
				}
				save.setEnabled(true);
				ok.setEnabled(true);
			}
		} else if( e.getSource() == ok ){
			okPressed = true;
			dispose(); 
		} else if( e.getSource() == cancel ){ // cancel
			dMap = null;
			dispose();
		}
	}
	
	public boolean okPressed() {
		return okPressed;
	}
	
	public Map<double[], Map<double[], Double>> getDistanceMap() {
		return dMap;
	}
}