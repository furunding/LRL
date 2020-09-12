# 仿真服务器ip
sim_server_ip = '127.0.0.1'
# 基础容器映射端口
base_map_port = 8100
# 镜像名
image_name = 'sim_fast:1.3'

# 容器名前缀
docker_name_prefix = 'fu_env_'
# 容器里面想定文件绝对路径
scene_name = '/home/Joint_Operation_scenario.ntedt'
# 容器管理的脚本manage_client所在路径(这里给的相对路径)
# 'prefix': './',                          

# 挂载点
volume_list = []
# 'max_game_len': 350             # 最大决策次数
## 记录回放相关设置
# 是否记录
save_replay = False
# 回放保存路径       
replay_dir = './replays'