# 1. 가상 디스크 파일 생성 (Team A: 500GB, Team B: 1TB)
sudo mkdir -p /data
sudo dd if=/dev/zero of=/data/teamA_disk.img bs=1G count=500
sudo dd if=/dev/zero of=/data/teamB_disk.img bs=1G count=1000

# 2. 파일 시스템 포맷 (ext4)
sudo mkfs.ext4 /data/teamA_disk.img
sudo mkfs.ext4 /data/teamB_disk.img

# 3. 마운트할 폴더 생성
sudo mkdir -p /mnt/teamA /mnt/teamB

# 4. 재부팅 시 자동 마운트되도록 /etc/fstab에 추가
echo "/data/teamA_disk.img /mnt/teamA ext4 loop 0 0" | sudo tee -a /etc/fstab
echo "/data/teamB_disk.img /mnt/teamB ext4 loop 0 0" | sudo tee -a /etc/fstab

# 5. 설정한 내용으로 즉시 마운트 실행
sudo mount -a