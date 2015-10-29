package spawnn.gui;

import java.awt.CardLayout;
import java.awt.Cursor;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
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

import net.miginfocom.swing.MigLayout;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Polygon;

public class DistMatrixDialog extends JDialog implements ActionListener {

	private static final long serialVersionUID = 2783445681242978907L;
	JButton save, cancel;
	
	JPanel cards;
	JComboBox cb, adjCb;
	JTextField power, knns, selFile;
	JCheckBox rowNorm;
	JButton btnDistMatrix;
	File file = null;

	public static enum DistMatType {InvDistance, kNN, Adjacency};
	
	public static enum AdjMode {Queen,Rook};
	
	private enum Card { one, two, three };
	
	private Dist<double[]> dist;
	private SpatialDataFrame sd;
	
	private Frame parent;
	
	public DistMatrixDialog(Frame parent, String string, boolean b, SpatialDataFrame sd, int[] ga ) {
		super(parent, string, b);
		boolean adjEnable = sd.geoms != null && ( sd.geoms.get(0) instanceof Polygon || sd.geoms.get(0) instanceof MultiPolygon );
		this.parent = parent;
		this.dist = new EuclideanDist(ga);
		this.sd = sd;
		
		setLayout(new MigLayout(""));

		add(new JLabel("Type:"));
		cb = new JComboBox();
		if( ga.length > 0 ) {
			cb.addItem(DistMatType.InvDistance);
			cb.addItem(DistMatType.kNN);
		}
		if( adjEnable )
			cb.addItem(DistMatType.Adjacency);
		cb.addActionListener(this);
		add(cb, "span 2, wrap");
		
		JPanel jp_1 = new JPanel(new MigLayout());
		jp_1.add(new JLabel("Power:"));
		power = new JTextField("1", 3);
		jp_1.add(power, "wrap");

		JPanel jp_2 = new JPanel(new MigLayout());
		jp_2.add(new JLabel("k:"));
		knns = new JTextField("5", 3);
		jp_2.add(knns, "wrap");
		
		JPanel jp_3 = new JPanel(new MigLayout());
		jp_3.add(new JLabel("Type:"));
		adjCb = new JComboBox();
		adjCb.setModel(new DefaultComboBoxModel(AdjMode.values()));
		jp_3.add(adjCb, "wrap");
		
		cards = new JPanel(new CardLayout());
		if( ga.length > 0 ) {
			cards.add(jp_1, Card.one.toString());
			cards.add(jp_2, Card.two.toString());
		}
		if( adjEnable )
			cards.add(jp_3, Card.three.toString());
		
		add(cards, "span 3, wrap");
		
		add( new JLabel("Row-normalize:"));
		rowNorm = new JCheckBox();
		rowNorm.setEnabled(true);
		add( rowNorm, "span 2, wrap" );
		
		add( new JLabel("Target file:"),""); 
		
		selFile = new JTextField();
		selFile.setColumns(20);
		selFile.setEditable(true);
		add(selFile,"growx");
		
		btnDistMatrix = new JButton("Select...");
		btnDistMatrix.addActionListener(this);
		add(btnDistMatrix, "wrap");

		save = new JButton("Create...");
		save.addActionListener(this);
		cancel = new JButton("Cancel");
		cancel.addActionListener(this);
		add(save, "span2, split 2, push");
		add(cancel);

		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		pack();
		setAlwaysOnTop(true);
		setLocationRelativeTo(parent);
		setVisible(true);
	}

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
		} else if (e.getSource() == save) {
			setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
			
			Map<double[], Map<double[], Double>> dMap = null;
			if( cb.getSelectedItem() == DistMatType.Adjacency ) {
				dMap = GeoUtils.listsToWeights( GeoUtils.getContiguityMap(sd.samples, sd.geoms, adjCb.getSelectedItem() == AdjMode.Rook ));
			} else if( cb.getSelectedItem() == DistMatType.InvDistance ) {
				dMap = GeoUtils.getInverseDistanceMatrix(sd.samples, dist, Double.parseDouble(power.getText() ) );
			} else { // knn
				dMap = GeoUtils.listsToWeights( GeoUtils.getKNNs(sd.samples, dist, Integer.parseInt(knns.getText())));
			}
			
			if( rowNorm.isSelected() )
				GeoUtils.rowNormalizeMatrix(dMap);
			GeoUtils.writeDistMatrixKeyValue(dMap, sd.samples, file );
			
			setCursor(Cursor.getDefaultCursor());
			dispose();
		} else if( e.getSource() == btnDistMatrix ) {
			JFileChooser fc = new JFileChooser();
			int state = fc.showSaveDialog(this);
			if( state == JFileChooser.APPROVE_OPTION ) { 
			      file = fc.getSelectedFile();
			      selFile.setText(""+file);
			}
		} else {
			dispose();
		}
	}
}
