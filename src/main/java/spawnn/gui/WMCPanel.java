package spawnn.gui;

import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.List;
import java.util.Map;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class WMCPanel extends JPanel implements ActionListener {

	private static final long serialVersionUID = 1671229973714048612L;
	private JTextField alphaField,betaField, infoField;
	private JButton selDM, createDM;
	
	private Frame parent;
	
	public WMCPanel(Frame parent) {
		this.parent = parent;
		setLayout(new MigLayout(""));
		
		JLabel lblAlpha = new JLabel("Alpha:");
		add(lblAlpha, "");
		
		alphaField = new JTextField();
		alphaField.setText("0.5");
		alphaField.setColumns(10);
		add(alphaField, "");
				
		JLabel lblBeta = new JLabel("Beta:");
		add(lblBeta, "");
		
		betaField = new JTextField();
		betaField.setText("0.5");
		betaField.setColumns(10);
		add(betaField, "wrap");
				
		add( new JLabel("Dist. matrix:"),""); 
		
		selDM = new JButton("Load...");
		selDM.addActionListener(this);
		add(selDM, "");
		
		createDM = new JButton("Create...");
		createDM.addActionListener(this);
		add(createDM, "skip, wrap");
		
		infoField = new JTextField("No dist. matrix loaded yet.");
		infoField.setEditable(false);
		add(infoField, "span 4, growx");
	}
	
	Map<double[],Map<double[],Double>> dMap = null;

	@Override
	public void actionPerformed(ActionEvent ae) {
		if( ae.getSource() == selDM ) {
			JFileChooser fc = new JFileChooser();
			int state = fc.showOpenDialog(this);
			if( state == JFileChooser.APPROVE_OPTION ) { 
			      File file = fc.getSelectedFile();
			      try {
			    	  dMap = GeoUtils.readDistMatrixKeyValue(normedSamples, file);
			      } catch( Exception e ) {
			    	  JOptionPane.showMessageDialog(this, "Could read/parse distance matrix file: "+e.getLocalizedMessage(), "Read/Parse error!", JOptionPane.ERROR_MESSAGE);
			    	  e.printStackTrace();
			      }
			}
		} else if( ae.getSource() == createDM ) {
			DistMatrixDialog dmd = new DistMatrixDialog(parent, "Create dist. matrix", true, normedSamples, sdf, gaAll);
			if( dmd.okPressed )
				dMap = dmd.getDistanceMap();
			else
				dMap = null;
		}
		if( dMap != null ) {
			int entries = 0;
			for( Map<double[], Double> s : dMap.values() )
				entries += s.size();
			infoField.setText(entries+" entries.");
		}
	}
				
	public Map<double[],Map<double[],Double>> getDistanceMap() {
		return dMap;
	}
	
	private List<double[]> normedSamples;
	private SpatialDataFrame sdf;
	private int[] gaAll;
	
	public void setTrainingData( List<double[]> normedSamples, SpatialDataFrame spatialData, int[] gaAll) {
		this.normedSamples = normedSamples;
		this.sdf = spatialData;
		this.gaAll = gaAll;
	}
	
	public double getAlpha() {
		return Double.parseDouble(alphaField.getText());
	}

	public double getBeta() {
		return Double.parseDouble(betaField.getText());
	}
}
