#!/bin/bash
# INCOGNITOVISION - RDK X5 Deployment Script
# Usage: ./deploy_rdk.sh <board-ip> [username]

BOARD_IP="${1:-192.168.1.100}"
USERNAME="${2:-sunrise}"

echo "================================================"
echo "INCOGNITOVISION - RDK X5 Deployment"
echo "================================================"
echo "Target: $USERNAME@$BOARD_IP"
echo ""

# Check SSH connectivity
echo "[1/5] Testing SSH connection..."
ssh -o ConnectTimeout=5 -o BatchMode=yes $USERNAME@$BOARD_IP "echo 'SSH OK'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Cannot connect to $BOARD_IP"
    echo "Please ensure:"
    echo "  - RDK X5 is powered on"
    echo "  - Connected to the same network"
    echo "  - SSH is enabled on the board"
    echo ""
    echo "To find board IP, try: arp -a | grep sunrise"
    exit 1
fi

# Check camera
echo "[2/5] Checking camera..."
ssh $USERNAME@$BOARD_IP "ls /dev/video* 2>/dev/null || echo 'No cameras found'"

# Clone/update repository
echo "[3/5] Cloning repository..."
ssh $USERNAME@$BOARD_IP << 'EOF'
    if [ -d "securitycam" ]; then
        cd securitycam && git pull
    else
        git clone https://github.com/lemonfungus/securitycam-py-repo.git securitycam
    fi
EOF

# Install Python dependencies
echo "[4/5] Installing AI service dependencies..."
ssh $USERNAME@$BOARD_IP << 'EOF'
    cd securitycam/ai_service
    pip3 install -r requirements.txt --user
EOF

# Start AI service
echo "[5/5] Starting AI service..."
echo "To run the AI server on the board:"
echo ""
echo "  ssh $USERNAME@$BOARD_IP"
echo "  cd securitycam/ai_service"
echo "  export CAMERA_SOURCE=0"
echo "  python3 ai_server.py"
echo ""
echo "================================================"
echo "Deployment complete!"
echo "Access the video feed at: http://$BOARD_IP:5000/video_feed"
echo "================================================"
